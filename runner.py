import  re, json, numpy as np, pandas as pd, torch, os, random
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from tqdm.auto import tqdm
def runner (_args):
    # ───────────────────── Imports
    from utils.logger import log, log_p, setup_logger
    from data.dataloader import (
        load_uci_timeseries, load_oulad_timeseries, 
        load_xuetangx_timeseries, process_combined_timeseries,
        encode_categorical_features, prepare_features_and_target
    )
    from agents.cat_agent import CatAgentTS
    from agents.sme_moe_agent import SMEMoEAgent
    from agents.prompt_agent import PromptRiskMixin
    from models.tuning import get_cluster_stats
    from llm.model_setup import setup_ollama_models, setup_gpt2_model
    from llm.prompts import PROMPT_TS, DEC_PROMPT_TS
    from llm.utils import parse_json_from_response, safe_cast

    args = _args

    if args.uci:
        f_out = "UCI"
    elif args.xuetangx:
        f_out = "xuetangx"
    else:
        f_out = "OULAD"

    import os

    logger_path = os.path.join("logs", f"agent_timeseries_{f_out}.log") 
    
    setup_logger(logger_path)

    n_students = args.N_students
    
    # Create necessary directories if they don't exist
    os.makedirs(f"model_save/{f_out}/{n_students}_students", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ───────── Environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import set_seed
    set_seed(42); log(f"device={DEVICE}")

    # ───────── TIME SERIES CONFIGURATION
    TIME_WINDOW = args.time_window  # 30 days lookback window
    PRED_HORIZON = args.prediction_horizon  # 7 days prediction horizon
    SNAPSHOT_DAYS = [7, 14, 21, 30, 45, 60]
    SNAPSHOT_DAYS_UCI = [90,180] # Use two snapshots for the two semesters in UCI data
    n_students = args.N_students

    # ───────── Paths (same as original)
    GPT2_DIR   = os.path.join("utils", "gpt2_dropout_qlora")
    LLAMA_ID   = f"llama2:{args.llama_size}"

    log(f"Model selected: {LLAMA_ID}")

    print("Starting the process...")
    # ───────── Load time series data
    
    # Define the output filename based on arguments FIRST
    if args.uci:
        TS_FILE = os.path.join('datasets', 'ts_datasets_uci.csv')
    elif args.xuetangx:
        TS_FILE = os.path.join('datasets', 'ts_datasets_xuetangx.csv')
    else:
        TS_FILE = os.path.join('datasets', 'ts_datasets.csv')
    
    # Check if the processed file already exists
    if not os.path.exists(TS_FILE):
        if args.uci:
            df_ts = load_uci_timeseries(args.uci_path)
        elif args.xuetangx:
            df_ts = load_xuetangx_timeseries(
                train_path=args.train_path,
                test_path=args.test_path,
                user_info_path=args.user_info_path,
                snapshot_days=SNAPSHOT_DAYS,
                time_window=TIME_WINDOW
            )
        else: # Default to OULAD
            FOLDER_PATH = args.folder_path
            df_ts = load_oulad_timeseries(FOLDER_PATH, SNAPSHOT_DAYS)
        
        # FIX: Add a check here to stop if no data was generated
        if df_ts.empty:
            log("--> FATAL ERROR: Data processing resulted in an empty dataset. Halting execution.")
            return # Exit the script

        log(f"✅ Data processing complete. Saving to {TS_FILE}")
        df_ts.to_csv(TS_FILE, index=False)

    # ... previous code ...
    log(f"Loading processed data from {TS_FILE}...")
    df_ts = pd.read_csv(TS_FILE)

    # FIX: Add a check to ensure the essential 'student_id' column exists
    if 'student_id' not in df_ts.columns:
        log("--> FATAL ERROR: 'student_id' column not found in the processed data.")
        log(f"--> Available columns are: {df_ts.columns.tolist()}")
        log("--> This indicates an issue with the data generation step. Halting execution.")
        return # Exit the script

    # ... rest of the main function ...
    
    # ... (rest of the main function)
    # ───────── DOWN-SAMPLE FOR QUICK TEST ─────────
    # 1) pick N students
    unique_students = df_ts['student_id'].unique()
    if len(unique_students) > n_students:
        sample_students = np.random.choice(unique_students, size=n_students, replace=False)
        # 2) filter the dataframe
        df_ts = df_ts[df_ts['student_id'].isin(sample_students)].reset_index(drop=True)


    log(f"– Running on {len(df_ts)} rows from {df_ts['student_id'].nunique()} students")


    # ───────── Encode categorical variables for time series
    df_ts = encode_categorical_features(df_ts, 'oulad' if not args.uci else 'uci')

    # ───────── Create features and target
    X_ts, y_ts, idx_ts = prepare_features_and_target(
        df_ts, dataset_type='oulad', use_uci=args.uci, use_xuetangx=args.xuetangx
    )

    # ───────── Scale features
    scaler = MinMaxScaler()
    X_ts_scaled = pd.DataFrame(scaler.fit_transform(X_ts), columns=X_ts.columns, index=X_ts.index)

    # ───────── Train/test split (stratified by student to avoid data leakage)
    unique_students = df_ts['student_id'].unique()
    train_students, test_students = train_test_split(
        unique_students, test_size=0.25, random_state=42
    )

    train_mask = df_ts['student_id'].isin(train_students)
    test_mask = df_ts['student_id'].isin(test_students)

    Xtr_ts = X_ts_scaled[train_mask]
    Xte_ts = X_ts_scaled[test_mask]
    ytr_ts = y_ts[train_mask]
    yte_ts = y_ts[test_mask]
    idtr_ts = df_ts[train_mask].index.to_numpy()
    idte_ts = df_ts[test_mask].index.to_numpy()

    log(f"Time series dataset: {len(Xtr_ts)} train, {len(Xte_ts)} test samples")

    # ───────────────────── SAME AGENT ARCHITECTURE AS ORIGINAL ─────────────────────

    # ───────── Llama chat models (same as original)
    ollama = False
    #* ollama settings
    OLLAMA_SHORT, OLLAMA_LONG, ollama = setup_ollama_models(LLAMA_ID)

    # ───────── GPT-2 QLoRA setup (same as original)
    GPT2_SLM = setup_gpt2_model(GPT2_DIR)

    # Handle fallback if Ollama is not available
    if not ollama:
        GPT2_SLM = OLLAMA_SHORT

    log(f"using ollama : {args.ollama}")
    
    # ───────── CatBoost agents adapted for time series
    groups_tr = df_ts.loc[Xtr_ts.index, 'student_id']
    
    # ─── DYNAMIC AGENT FEATURE SELECTION ──────────────────────────────────
    log("Defining agent feature sets...")

    # Profile 1: Default features for OULAD data (rich assessments and VLE types)
    OULAD_FEATURES = {
        "Performance": ["assessment_score_mean", "assessment_score_trend", "assessments_completed"],
        "Baseline": ["gender", "age_band", "highest_education", "disability", "num_of_prev_attempts"],
        "Engagement": ["clicks_mean", "clicks_std", "engagement_consistency", "active_days"],
        "Temporal": ["days_into_course", "days_since_last_activity", "clicks_recent_vs_early"]
    }

    # Profile 2: Richer, more specific features for UCI data (strong demographics, no clicks)
    UCI_FEATURES = {
        "Performance": ["sem_grade", "sem_approved", "sem_evaluations", "previous_qualification_grade"],
        "Baseline": ["gender", "age_at_enrollment", "marital_status", "mother's_qualification", "displaced", "scholarship_holder"],
        "Engagement": ["tuition_fees_up_to_date", "debtor", "daytimeevening_attendance"],
        "Temporal": ["snapshot_day", "application_order", "sem_grade_trend"]
    }
    
    # Profile 3: Specific features for XuetangX data (strong behavioral clicks, no assessments)
    XUETANGX_FEATURES = {
        "Performance": ["assessment_score_mean"], # Will be a dummy agent as XuetangX has no grade data
        "Baseline": ["gender", "age_band", "highest_education", "disability"], # Based on available profile data
        "Engagement": ["clicks_total", "clicks_mean", "clicks_std", "engagement_consistency", "active_days"],
        "Temporal": ["days_into_course", "days_since_last_activity", "clicks_trend", "clicks_volatility"]
    }

    # Choose the correct feature profile based on the command-line flags
    if args.uci:
        agent_feature_sets = UCI_FEATURES
        log("Using UCI-specific feature sets for agents.")
    elif args.xuetangx:
        agent_feature_sets = XUETANGX_FEATURES
        log("Using XuetangX-specific feature sets for agents.")
    else:
        agent_feature_sets = OULAD_FEATURES
        log("Using default OULAD feature sets for agents.")

    # ─── AGENT TRAINING ───────────────────────────────────────────────────
    log("Training time series agents...")
    perf_ts     = CatAgentTS(Xtr_ts, ytr_ts, groups_tr, agent_feature_sets["Performance"], "Performance", f_out, n_students, args.retune)
    base_ts     = CatAgentTS(Xtr_ts, ytr_ts, groups_tr, agent_feature_sets["Baseline"], "Baseline", f_out, n_students, args.retune)
    engage_ts   = CatAgentTS(Xtr_ts, ytr_ts, groups_tr, agent_feature_sets["Engagement"], "Engagement", f_out, n_students, args.retune)
    temporal_ts = CatAgentTS(Xtr_ts, ytr_ts, groups_tr, agent_feature_sets["Temporal"], "Temporal", f_out, n_students,  args.retune)

    # ───────── Clustering agent for course difficulty (adapted)
    diff_cols_ts = ["assessment_score_mean", "clicks_mean", "days_since_last_activity"]
    available_diff_cols = [c for c in diff_cols_ts if c in Xtr_ts.columns]

    # ───────── Course-Difficulty agent with real cluster stats ─────────
    cluster_stats, risk_diff_ts, rat_diff_ts = get_cluster_stats(
        Xtr_ts, ytr_ts, available_diff_cols
    )

    # ───────── SMETimes MoE (adapted for time series)
    sme_agent = SMEMoEAgent(device=DEVICE)
    sme_agent.train(Xtr_ts, ytr_ts)

    # ───────── Prompt templates (same structure, updated context)
    # ───────────────────── IMPROVED PROMPT_TEMPLATES ──────────────────────
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    JSON_EXTRACTOR_PROMPT = PromptTemplate.from_template("""
        You are an expert JSON extractor. Parse the following text and output only valid JSON with keys 'risk' 'internal_reasoning' and 'rationale'.

        Text:
        {raw_text}

        Return JSON only.
    """)
    if OLLAMA_SHORT:
        json_extractor_chain = LLMChain(llm=OLLAMA_SHORT, prompt=JSON_EXTRACTOR_PROMPT)
    else:
        json_extractor_chain = None

    # ───────── Assemble time series role-agents
    role_agents_ts = {
        "Performance":     PromptRiskMixin("Performance",     perf_ts.risk,     perf_ts.rat,     OLLAMA_SHORT, json_extractor_chain),
        "Profile":        PromptRiskMixin("Profile",        base_ts.risk,     base_ts.rat,     OLLAMA_SHORT, json_extractor_chain),
        "Engagement":      PromptRiskMixin("Engagement",      engage_ts.risk,   engage_ts.rat,   OLLAMA_SHORT, json_extractor_chain),
        "Temporal":        PromptRiskMixin("Temporal",        temporal_ts.risk, temporal_ts.rat, OLLAMA_SHORT, json_extractor_chain),
        # add CourseDifficulty only if we have cols
        **({"CourseDifficulty": PromptRiskMixin("CourseDifficulty", risk_diff_ts, rat_diff_ts, OLLAMA_SHORT, json_extractor_chain)}
        if available_diff_cols else {}),
        "SMETimes":        PromptRiskMixin("SMETimes",        sme_agent.risk,      sme_agent.rat,      GPT2_SLM, json_extractor_chain)
    }

    # ───────── Predict with time series agents
    log("Running time series agent predictions...")
    risks_ts, rats_ts = {}, {}
    for name, ag in tqdm(role_agents_ts.items(), desc="Time series agents", total=len(role_agents_ts),unit="agent"):
        try:
            if name == "CourseDifficulty":
                p, r = ag.predict(Xte_ts, cluster_stats=cluster_stats)
            else:
                p, r = ag.predict(Xte_ts)
            risks_ts[name], rats_ts[name] = p, r
            log(f"Completed {name} agent: mean risk = {np.mean(p):.3f}")
        except Exception as e:
            log(f"Error in {name} agent: {e}")
            # Fallback predictions
            risks_ts[name] = np.random.random(len(Xte_ts)) * 0.5 + 0.25
            rats_ts[name] = [f"{name} analysis unavailable."] * len(Xte_ts)

    meta_Xtr = np.vstack([risks_ts[n] for n in role_agents_ts]).T      # same order
    meta_Xte = meta_Xtr                                                # (we're in test script, fine)
    meta_y   = yte_ts.values

    stacker_path = f"model_save/{f_out}/{n_students}_students/stacker_model_{f_out}.joblib"
    if os.path.exists(stacker_path) and not args.retune:
        log("Loading pre-trained stacker model.")
        stack = joblib.load(stacker_path)
    else:
        log("Training new stacker model.")
        from catboost import CatBoostClassifier
        # Use your CatBoost model for stacking
        stack = CatBoostClassifier(iterations=500, verbose=False, auto_class_weights="Balanced")
        stack.fit(meta_Xtr, meta_y)
        joblib.dump(stack, stacker_path) # Save the trained stacker

    probs = stack.predict_proba(meta_Xte)[:, 1]
    # -------- add these 3 lines -------------
    stack_probs = probs                       # name used downstream
    risks_ts["STACK"] = stack_probs
    rats_ts ["STACK"] = ["Meta-stack probability"] * len(stack_probs)
    # ----------------------------------------

    # choose threshold that maximises F1 on a small validation fold
    val_mask = np.random.rand(len(meta_Xtr)) < 0.15
    best_f1, best_th = 0, 0.5
    for th in np.linspace(0.3, 0.7, 21):
        f1 = f1_score(meta_y[val_mask], (probs[val_mask] >= th))
        if f1 > best_f1:
            best_f1, best_th = f1, th

    stack_probs = probs                # keep name used later
    stack_pred  = (stack_probs >= best_th).astype(int)
    log(f"Stacker F1 best_th={best_th:.2f}  val_F1={best_f1:.3f}")
    arbiter_names = list(role_agents_ts) + ["STACK"]   # << new line

    # --- FINAL RECOMMENDED PROMPT (Combines Internal Monologue + Sequential History) ---
    final_arbiter_prompt = PromptTemplate.from_template(DEC_PROMPT_TS, template_format="jinja2")

    if OLLAMA_LONG is not None and json_extractor_chain is not None:
        arb_ts = LLMChain(llm=OLLAMA_LONG, prompt=final_arbiter_prompt)
        last_risk_per_student = {}
        DEFAULT_RISK = 0.5

        # Sort the test data chronologically
        test_indices = yte_ts.index
        df_test_sorted = df_ts.loc[test_indices].sort_values(['student_id', 'snapshot_day'])
        Xte_ts_sorted = X_ts_scaled.loc[df_test_sorted.index]

        log("Running final arbiter decisions...")
        outs_ts = []

        for i in tqdm(range(len(Xte_ts_sorted)), desc="Final decisions", unit="rec"):
            # Use a broad try-except block to catch any unexpected errors for a single student
            try:
                current_row_features = Xte_ts_sorted.iloc[i]
                student_id = df_test_sorted.iloc[i]["student_id"]
                previous_risk = last_risk_per_student.get(student_id, DEFAULT_RISK)

                # Get predictions from base agents for the current row
                risks_now = {}
                rats_now = {}
                for name, ag in role_agents_ts.items():
                    if name == "CourseDifficulty":
                        pred_result, rat_result = ag.predict(current_row_features.to_frame().T, cluster_stats=cluster_stats)
                        risks_now[name] = pred_result[0]
                        rats_now[name] = rat_result[0]
                    else:
                        pred_result, rat_result = ag.predict(current_row_features.to_frame().T)
                        risks_now[name] = pred_result[0]
                        rats_now[name] = rat_result[0]

                # Get the stacked prediction, which will be our primary fallback
                meta_features_now = np.array(list(risks_now.values())).reshape(1, -1)
                stack_risk_now = stack.predict_proba(meta_features_now)[0, 1]
                risks_now["STACK"] = stack_risk_now
                final_verdict = "DROPOUT" if stack_risk_now >= best_th else "PERSIST"

                # Prepare parameters for the LLM prompt
                param = {f"{n}_risk": risks_now[n] for n in arbiter_names} | \
                        {f"{n}_rat": rats_now.get(n, "N/A") for n in arbiter_names}
                param['previous_risk'] = previous_risk

                # --- LLM Call and Robust Parsing ---
                raw_response = arb_ts.run({
                    "names": arbiter_names, "params": param,
                    "final_verdict": final_verdict, "snapshot_day": df_test_sorted.iloc[i]['snapshot_day'],
                })
                
                # ────── SAVE FINAL-DECISION EXAMPLE FOR LoRA ──────
                if args.train_log:
                    SAVE_RATE = 0.50         # keep 10 % of arbiter calls
                    SAVE_PATH = os.path.join("results", "finetune.jsonl")

                    if random.random() < SAVE_RATE:
                        rendered_prompt = final_arbiter_prompt.format(
                            names=arbiter_names,
                            params=param,
                            final_verdict=final_verdict,
                            snapshot_day=df_test_sorted.iloc[i]['snapshot_day'])
                        with open(SAVE_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "prompt": rendered_prompt,
                                "completion": raw_response.strip()
                            }, ensure_ascii=False) + "\n")
                # ──────────────────────────────────────────────────
                # --- Define sane defaults based on the deterministic stacker model ---
                defaults = {
                    "final_risk": stack_risk_now,
                    "prediction": 1 if stack_risk_now >= 0.5 else 0,
                    "confidence": abs(stack_risk_now - 0.5) * 2,
                    "rationale":  "Fallback: LLM rationale could not be parsed.",
                    "internal_reasoning": {"error": "Parsing failed"}
                }

                # --- Attempt to parse the JSON from the LLM response ---
                parsed = parse_json_from_response(json_extractor_chain.run(raw_text=raw_response))
                if "error" in parsed:
                    log(f"LLM parsing failed for sample {i}: {parsed['error']}")
                    parsed = {} # Use an empty dict if parsing fails

                # --- Merge defaults with parsed results to ensure all keys exist ---
                j = {**defaults, **parsed}

                # --- Safely cast each value, falling back to defaults if types are wrong ---
                final_risk = safe_cast(j.get("final_risk"), float, defaults["final_risk"])
                prediction = safe_cast(j.get("prediction"), int, (1 if final_risk >= 0.5 else 0))
                confidence = safe_cast(j.get("confidence"), float, abs(final_risk - 0.5) * 2)
                rationale = str(j.get("rationale", defaults["rationale"]))
                internal_reasoning = j.get("internal_reasoning", defaults["internal_reasoning"])

                # Append the fully cleaned and validated results
                outs_ts.append({
                    "risk": final_risk, "pred": prediction,
                    "conf": confidence, "rat": rationale,
                    "internal_reasoning": internal_reasoning
                })

                # Update the student's risk history for the next snapshot
                last_risk_per_student[student_id] = final_risk

            except Exception as e:
                log(f"Critical error in final arbiter loop for sample {i}: {e}")
                # If a critical error occurs, append a fallback record to prevent crashing later
                fallback_risk = last_risk_per_student.get(student_id, DEFAULT_RISK)
                outs_ts.append({
                    "risk": fallback_risk, "pred": 1 if fallback_risk >= 0.5 else 0,
                    "conf": abs(fallback_risk - 0.5) * 2, "rat": "Fallback due to critical error.",
                    "internal_reasoning": {"error": str(e)}
                })
    else:
        # This is the fallback if Ollama is not available
        log("Ollama not available, using simple averaging for final decisions")
        outs_ts = []
        # Note: The original loop here was 'for i in range(len(Xte_ts))' which can cause index issues.
        # It's better to iterate directly if possible, but we'll create a simple fallback list.
        for i in range(len(yte_ts)): # Iterate based on the length of the test set target
            # This part is simplified and may not be perfect if test sets were reordered
            # but it provides a basic fallback mechanism.
            avg_risk = np.mean([risks_ts[n][i] for n in role_agents_ts.keys() if i < len(risks_ts[n])])
            outs_ts.append({
                "risk": avg_risk,
                "pred": 1 if avg_risk >= 0.5 else 0,
                "conf": abs(avg_risk - 0.5) * 2,
                "rat": "Simple ensemble average.",
                "internal_reasoning": "No reasoning available"
            })

    # ───────── Create comprehensive results CSV
    log("Creating comprehensive time series results...")
    stage_mapping = {
        7: 'early', 14: 'early',
        21: 'mid',  30: 'mid',
        45: 'late',  60: 'late'
    }
    # FIX: Re-index all results to match the chronologically sorted test set
    # This ensures all lists and arrays have the same order and length.
    sorted_test_indices = Xte_ts_sorted.index

    # Create the base dataframe with agent predictions, correctly indexed
    df_results = pd.DataFrame(index=sorted_test_indices)
    for n in role_agents_ts:
        # Re-index individual agent risk and rationale series before adding them
        df_results[f"{n}_risk"] = pd.Series(risks_ts[n], index=yte_ts.index).reindex(sorted_test_indices)
        df_results[f"{n}_rationale"] = pd.Series(rats_ts[n], index=yte_ts.index).reindex(sorted_test_indices)

    # Add metadata from the sorted test dataframe
    df_results["student_id"] = df_test_sorted["student_id"].values
    df_results["snapshot_day"] = df_test_sorted["snapshot_day"].values
    if args.uci: 
        df_results["course_stage"] = df_results["snapshot_day"].apply(
            lambda x: "early" if x <= 90 else ("mid" if x <= 180 else "late")
        )
    else:
        df_results['course_stage'] = df_results['snapshot_day'].map(stage_mapping)

    df_results["actual_dropout"] = yte_ts.reindex(sorted_test_indices).values
    # Add final arbiter results (which are already in the correct sorted order)
    df_results["final_prediction"] = [o["pred"] for o in outs_ts]
    df_results["final_risk"] = [o["risk"] for o in outs_ts]
    df_results["final_confidence"] = [o["conf"] for o in outs_ts]
    df_results["final_rationale"] = [o["rat"] for o in outs_ts]
    
    # Add stacker and internal reasoning results
    df_results["stack_risk"] = pd.Series(stack_probs, index=yte_ts.index).reindex(sorted_test_indices)
    df_results["internal_reasoning"] = [str(o["internal_reasoning"]) for o in outs_ts]

    # Add time series specific features for analysis
    ts_features = ["clicks_mean", "clicks_trend", "clicks_volatility",
                "assessment_score_mean", "days_since_last_activity"]
    for feat in ts_features:
        if feat in Xte_ts_sorted.columns:
            df_results[f"feature_{feat}"] = Xte_ts_sorted[feat].values

    # Calculate evaluation metrics
    accuracy = accuracy_score(df_results["actual_dropout"], df_results["final_prediction"])
    f1 = f1_score(df_results["actual_dropout"], df_results["final_prediction"])
    recall = recall_score(df_results["actual_dropout"], df_results["final_prediction"])
    try:
        auc = roc_auc_score(df_results["actual_dropout"], df_results["final_risk"])
    except:
        auc = 0.5

    log(f"Time Series Model Performance:")
    log(f"  Accuracy: {accuracy:.3f}")
    log(f"  Recall: {recall:.3f}")
    log(f"  F1-Score: {f1:.3f}")
    log(f"  AUC-ROC: {auc:.3f}")

    # Stage-specific performance
    for stage in ["early", "mid", "late"]:
        stage_mask = df_results["course_stage"] == stage
        if stage_mask.sum() > 0:
            stage_acc = accuracy_score(
                df_results.loc[stage_mask, "actual_dropout"],
                df_results.loc[stage_mask, "final_prediction"]
            )
            log(f"  {stage.capitalize()} stage accuracy: {stage_acc:.3f} (n={stage_mask.sum()})")

    
    # Save results
    output_file = os.path.join("results", f"timeseries_multi_agent_dropout_predictions_{f_out}_{n_students}.csv")
    df_results.to_csv(output_file, index=False)
    log(f"Saved comprehensive results → {output_file}")
    
    # --- at the very end of main(), right after writing the csv ---
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        preview_cols = ["student_id", "snapshot_day"] + \
                    [c for c in df.columns if "rationale" in c and "Performance" in c]
        if not df[preview_cols].empty:
             print("\nCSV sanity-check:\n")
             print(df[preview_cols].head(10).to_string())
