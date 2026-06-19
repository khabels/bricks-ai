#![allow(clippy::too_many_arguments)]

use bricks_ai_core::*;
use bricks_ai_runtime::RuntimeModel;
use dotenvy::dotenv;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn main() {
    if let Err(error) = real_main() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(String::as_str).unwrap_or("help");

    match command {
        "demo" => run_demo(),
        "device" => run_device_diagnostics(),
        "pack" => run_pack_preview(),
        "providers" => run_provider_diagnostics(),
        "ollama-test" => run_ollama_diagnostics(),
        "memory" => run_memory_diagnostics(),
        "pack-stats" => run_knowledge_pack_stats(&args[2..]),
        "train-pack" | "teacher" => run_train_pack_entry(&args[2..]),
        "export" => run_export_demo(),
        "inspect-model" => run_inspect_model(&args[2..]),
        "predict" => run_runtime_predict(&args[2..]),
        "export-runtime-model" => run_export_runtime_model(&args[2..]),
        _ => {
            print_help();
            Ok(())
        }
    }
}

fn print_help() {
    println!("Bricks AI");
    println!();
    println!("Commands:");
    println!("  demo                 Run a small local grid/correlation training demo");
    println!("  device               Detect local GPU and show CPU fallback behavior");
    println!("  pack                 Preview universal training-pack prompts without API calls");
    println!("  providers            Detect Ollama/API providers available in .env");
    println!("  ollama-test          Test Ollama /api/tags and /api/generate");
    println!("  memory               Show CPU-memory profile and estimated trainer memory");
    println!("  pack-stats           Inspect local knowledge packs built by training");
    println!("  train-pack           Train locally from knowledge packs + Ollama/API enrichment");
    println!("                       Supports --resume, --extend, parallel dimensions, benchmark reports, RPM limits and fallback");
    println!("  export               Export a small engraved model JSON file");
    println!("  inspect-model        Inspect an engraved/runtime model JSON file");
    println!("  predict              Run a lightweight runtime prediction from engraved_model.json");
    println!("  export-runtime-model Normalize engraved_model.json into runtime_model.json");
    println!();
    println!("Examples:");
    println!("  cargo run -- device");
    println!("  cargo run -- providers");
    println!("  cargo run -- ollama-test");
    println!("  cargo run -- train-pack --steps 500 --items 8 --depth 2");
    println!("  cargo run -- train-pack --resume --steps 500");
    println!("  cargo run -- inspect-model --model engraved_model.json");
    println!("  cargo run -- predict --model engraved_model.json --input \"hello bricks\"");
}


fn run_inspect_model(args: &[String]) -> Result<(), Box<dyn Error>> {
    let model_path = flag_value(args, "--model").unwrap_or_else(|| "engraved_model.json".to_string());
    let runtime = RuntimeModel::load_from_file(&model_path)?;
    let summary = runtime.summary();
    println!("Bricks AI runtime model inspection");
    println!("model_path={}", model_path);
    println!("cases={}", summary.cases);
    println!("correlations={}", summary.correlations);
    println!("dimensions={}", runtime.model.dimensions.len());
    println!("dimension_paths={}", runtime.model.dimension_paths.len());
    println!("convergence_winner_history={}", runtime.model.convergence_cube.winner_history.len());
    println!("pre_final_destruction_runs={}", runtime.model.pre_final_destruction.runs);
    println!("pre_final_candidates_seen={}", runtime.model.pre_final_destruction.candidates_seen);
    println!("pre_final_candidates_forwarded={}", runtime.model.pre_final_destruction.candidates_forwarded);
    println!("pre_final_candidates_destroyed={}", runtime.model.pre_final_destruction.candidates_destroyed);
    println!("pre_final_cases_destroyed={}", runtime.model.pre_final_destruction.cases_destroyed);
    println!("pre_final_correlations_destroyed={}", runtime.model.pre_final_destruction.correlations_destroyed);
    if let Some(winner) = &runtime.model.convergence_cube.last_winner {
        println!("last_convergence_score={:.6}", winner.convergence_score);
        println!("last_convergence_final={}:{}:{}", winner.final_node.grid, winner.final_node.page, winner.final_node.case);
    }
    println!("input_node={:?}", summary.input_node);
    println!("output_node={:?}", summary.output_node);
    println!("avg_case_confidence={:.6}", summary.average_case_confidence);
    println!("avg_correlation_score={:.6}", summary.average_correlation_score);
    Ok(())
}

fn run_runtime_predict(args: &[String]) -> Result<(), Box<dyn Error>> {
    let model_path = flag_value(args, "--model").unwrap_or_else(|| "engraved_model.json".to_string());
    let input = flag_value(args, "--input").unwrap_or_else(|| "Bricks AI runtime probe".to_string());
    let passes = flag_usize(args, "--passes").unwrap_or(3);
    let runtime = RuntimeModel::load_from_file(&model_path)?;
    let prediction = runtime.predict_text(&input, passes);
    println!("Bricks AI runtime prediction");
    println!("model_path={}", model_path);
    println!("input_text={}", prediction.input_text);
    println!("input_signal={:.6}", prediction.input_signal);
    println!("output_node={:?}", prediction.output_node);
    println!("output_signal={:.6}", prediction.output_signal);
    println!("passes={}", prediction.passes);
    println!("activated_nodes={}", prediction.activated_nodes);
    Ok(())
}

fn run_export_runtime_model(args: &[String]) -> Result<(), Box<dyn Error>> {
    let input_path = flag_value(args, "--model").unwrap_or_else(|| "engraved_model.json".to_string());
    let output_path = flag_value(args, "--out").unwrap_or_else(|| "runtime_model.json".to_string());
    let runtime = RuntimeModel::load_from_file(&input_path)?;
    runtime.save_runtime_json(&output_path)?;
    println!("runtime model exported");
    println!("input={}", input_path);
    println!("output={}", output_path);
    Ok(())
}

fn run_demo() -> Result<(), Box<dyn Error>> {
    let mut trainer = new_configured_trainer();
    let (_input, _output) = configure_demo_graph(&mut trainer);

    let (input, output) = demo_input_output_nodes(&trainer);

    for epoch in 0..100 {
        let loss = trainer.train_step(&[(input, 1.0)], &[(output, 1.0)], 1);
        if epoch % 10 == 0 {
            println!("epoch={epoch}, loss={loss:.6}");
        }
        if epoch % 25 == 0 {
            trainer.prune_weak_links(0.001);
        }
    }

    trainer.engrave_validated_weights(0.01, 0.01);
    let prediction = trainer.predict(&[(input, 1.0)], &[output], 1);

    println!("final prediction grid {} / page {} / case {} = {:.4}", output.grid, output.page, output.case, prediction[0]);
    println!("engraved cases = {}", trainer.export_engraved_model().cases.len());
    println!("active correlations = {}", trainer.correlations.iter().filter(|c| c.active).count());
    Ok(())
}

fn run_pack_preview() -> Result<(), Box<dyn Error>> {
    let mut trainer = RawAITrainer::new(2, 64);
    trainer.pack_state.max_items_per_theme = 5;
    trainer.training_pack.max_depth = 1;
    trainer.start_pack_training();

    for step in 0..5 {
        match trainer.pack_training_tick() {
            PackTickAction::NeedSubthemeExpansion { job, prompt } => {
                println!("--- step {step}: expand {:?} ---", job.theme_path);
                println!("{}", first_lines(&prompt, 8));

                trainer.accept_generated_subthemes(&job, vec![
                    GeneratedSubTheme { name: "Foundations".to_string(), safety: job.safety },
                    GeneratedSubTheme { name: "Applications".to_string(), safety: job.safety },
                ]);
            }
            PackTickAction::NeedTrainingData { job, prompt } => {
                println!("--- step {step}: generate data {:?} ---", job.theme_path);
                println!("{}", first_lines(&prompt, 8));
            }
            PackTickAction::Finished => {
                println!("Pack finished.");
                break;
            }
            PackTickAction::Idle => {
                println!("Pack is idle or paused.");
                break;
            }
        }
    }

    println!("queue remaining = {}", trainer.pack_state.queue.len());
    Ok(())
}

fn run_provider_diagnostics() -> Result<(), Box<dyn Error>> {
    let device = detect_local_compute();
    print_compute_summary(&device);
    println!();
    let pool = TeacherProviderPool::from_env();

    println!("Bricks AI provider diagnostics");
    println!();

    if pool.providers.is_empty() {
        println!("No usable provider key found.");
        println!("Add at least one key in .env, for example OPENAI_API_KEY=...");
        return Ok(());
    }

    println!("Provider mode: {:?}", pool.mode);
    println!("Detected usable providers:");
    for provider in &pool.providers {
        println!(
            "- {} | model={} | env={} | endpoint={} | min_delay={}s | rpm={} | cooldown={}s",
            provider.display_name(),
            provider.model,
            provider.key_env,
            provider.endpoint,
            provider.min_interval.as_secs(),
            provider.requests_per_minute,
            provider.rate_limit_cooldown.as_secs()
        );
    }

    println!();
    println!("Default mode is local-first: Ollama is tried first on every request when enabled.");
    println!("Set BRICKS_AI_PROVIDER_MODE=ollama-only to forbid paid cloud fallback.");
    println!("Set BRICKS_AI_PROVIDER_MODE=cloud-only to ignore Ollama and use only API providers.");
    println!("429/rate-limit errors put that provider in cooldown and Bricks AI switches to the next provider unless mode is ollama-only.");
    Ok(())
}


fn run_ollama_diagnostics() -> Result<(), Box<dyn Error>> {
    let http = build_ollama_http_client();
    let base_url = env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2:latest".to_string());
    let tags_url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let generate_url = format!("{}/api/generate", base_url.trim_end_matches('/'));

    println!("Bricks AI Ollama diagnostics");
    println!("base_url={base_url}");
    println!("model={model}");
    println!("timeout_seconds={}", env_usize("OLLAMA_TIMEOUT_SECONDS").unwrap_or(300));
    println!("json_format={}", env_bool("OLLAMA_JSON_FORMAT").unwrap_or(false));
    println!("num_predict={}", env_usize("OLLAMA_NUM_PREDICT").unwrap_or(256));

    let tags_response = http
        .get(&tags_url)
        .send()
        .map_err(|error| boxed_error(format!("Ollama /api/tags connection failed: {error:?}")))?;
    let tags_status = tags_response.status();
    let tags_body = tags_response.text()?;
    println!("/api/tags status={tags_status}");
    println!("{}", first_chars(&tags_body, 600));

    if !tags_status.is_success() {
        return Err(boxed_error("Ollama /api/tags did not return success."));
    }

    let payload = build_ollama_payload(&model, "Return only this JSON object: {\"ok\":true}");
    let generate_response = http
        .post(&generate_url)
        .json(&payload)
        .send()
        .map_err(|error| boxed_error(format!("Ollama /api/generate connection failed: {error:?}")))?;
    let generate_status = generate_response.status();
    let generate_body = generate_response.text()?;
    println!("/api/generate status={generate_status}");
    println!("{}", first_chars(&generate_body, 1000));

    if !generate_status.is_success() {
        return Err(boxed_error("Ollama /api/generate did not return success."));
    }

    println!("Ollama test passed. Bricks AI can use Ollama.");
    Ok(())
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ComputePreference {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComputeInfo {
    preference: ComputePreference,
    effective_gpu: bool,
    gpu_name: Option<String>,
    reason: String,
}



#[derive(Debug, Clone, Serialize)]
pub struct KnowledgePackConfig {
    pub enabled: bool,
    pub mode: String,
    pub dir: String,
    pub train_from_pack_first: bool,
    pub save_generated_items: bool,
    pub max_reuse_per_item: usize,
    pub min_items_per_domain: usize,
    pub min_topics: usize,
    pub min_subtopics: usize,
    pub min_concepts: usize,
    pub build_depth: usize,
    pub reject_tiny_packs: bool,
    pub require_coverage: bool,
    pub extraction_items_per_concept: usize,
    pub build_chunked: bool,
    pub chunk_subtopics_per_topic: usize,
    pub chunk_concepts_per_subtopic: usize,
    pub max_build_chunks: usize,
    pub allow_live_fallback: bool,
    pub local_bootstrap_first: bool,
    pub provider_enrichment: bool,
}

impl KnowledgePackConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("BRICKS_AI_KNOWLEDGE_PACKS_ENABLED").unwrap_or(true),
            mode: env::var("BRICKS_AI_KNOWLEDGE_PACK_MODE").unwrap_or_else(|_| "hybrid".to_string()).to_ascii_lowercase(),
            dir: env::var("BRICKS_AI_KNOWLEDGE_PACK_DIR").unwrap_or_else(|_| "knowledge_packs".to_string()),
            train_from_pack_first: env_bool("BRICKS_AI_KNOWLEDGE_PACK_TRAIN_FIRST").unwrap_or(true),
            save_generated_items: env_bool("BRICKS_AI_KNOWLEDGE_PACK_SAVE_GENERATED").unwrap_or(true),
            max_reuse_per_item: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MAX_REUSE_PER_ITEM").unwrap_or(12).max(1),
            min_items_per_domain: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MIN_ITEMS_PER_DOMAIN").unwrap_or(96).max(8),
            min_topics: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MIN_TOPICS").unwrap_or(8).max(1),
            min_subtopics: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MIN_SUBTOPICS").unwrap_or(24).max(1),
            min_concepts: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MIN_CONCEPTS").unwrap_or(48).max(1),
            build_depth: env_usize("BRICKS_AI_KNOWLEDGE_PACK_BUILD_DEPTH").unwrap_or(3).max(1),
            reject_tiny_packs: env_bool("BRICKS_AI_KNOWLEDGE_PACK_REJECT_TINY_PACKS").unwrap_or(true),
            require_coverage: env_bool("BRICKS_AI_KNOWLEDGE_PACK_REQUIRE_COVERAGE").unwrap_or(true),
            extraction_items_per_concept: env_usize("BRICKS_AI_KNOWLEDGE_PACK_EXTRACTION_ITEMS_PER_CONCEPT").unwrap_or(3).max(1),
            build_chunked: env_bool("BRICKS_AI_KNOWLEDGE_PACK_BUILD_CHUNKED").unwrap_or(true),
            chunk_subtopics_per_topic: env_usize("BRICKS_AI_KNOWLEDGE_PACK_CHUNK_SUBTOPICS_PER_TOPIC").unwrap_or(3).max(1),
            chunk_concepts_per_subtopic: env_usize("BRICKS_AI_KNOWLEDGE_PACK_CHUNK_CONCEPTS_PER_SUBTOPIC").unwrap_or(2).max(1),
            max_build_chunks: env_usize("BRICKS_AI_KNOWLEDGE_PACK_MAX_BUILD_CHUNKS").unwrap_or(8).max(1),
            allow_live_fallback: env_bool("BRICKS_AI_KNOWLEDGE_PACK_ALLOW_LIVE_FALLBACK").unwrap_or(false),
            local_bootstrap_first: env_bool("BRICKS_AI_KNOWLEDGE_PACK_LOCAL_BOOTSTRAP_FIRST").unwrap_or(true),
            provider_enrichment: env_bool("BRICKS_AI_KNOWLEDGE_PACK_PROVIDER_ENRICHMENT").unwrap_or(false),
        }
    }

    fn continuous_only(&self) -> bool {
        !self.enabled || self.mode == "continuous-only" || self.mode == "live-only"
    }

    fn pack_only(&self) -> bool {
        self.enabled && self.mode == "pack-only"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KnowledgePack {
    #[serde(default)]
    format: String,
    #[serde(default)]
    version: u32,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    created_unix_ms: u64,
    #[serde(default)]
    updated_unix_ms: u64,
    #[serde(default)]
    coverage_score: f32,
    #[serde(default)]
    source_provider: String,
    #[serde(default)]
    curriculum_notes: String,
    #[serde(default)]
    topics: Vec<KnowledgeTopic>,
    #[serde(default)]
    items: Vec<KnowledgePackItem>,
}

impl KnowledgePack {
    fn new(domain: &str) -> Self {
        let now = unix_ms_now();
        Self {
            format: "bricks-ai-knowledge-pack".to_string(),
            version: 2,
            domain: domain.to_string(),
            created_unix_ms: now,
            updated_unix_ms: now,
            coverage_score: 0.0,
            source_provider: String::new(),
            curriculum_notes: String::new(),
            topics: Vec::new(),
            items: Vec::new(),
        }
    }

    fn accepted_count(&self) -> usize {
        self.items.iter().filter(|item| item.accepted).count()
    }

    fn rejected_count(&self) -> usize {
        self.items.iter().filter(|item| !item.accepted).count()
    }

    fn topic_count(&self) -> usize {
        self.topics.len()
    }

    fn subtopic_count(&self) -> usize {
        self.topics.iter().map(|topic| topic.subtopics.len()).sum()
    }

    fn concept_count(&self) -> usize {
        self.topics
            .iter()
            .flat_map(|topic| topic.subtopics.iter())
            .map(|subtopic| subtopic.concepts.len())
            .sum()
    }

    fn fact_count(&self) -> usize {
        self.topics
            .iter()
            .flat_map(|topic| topic.subtopics.iter())
            .flat_map(|subtopic| subtopic.concepts.iter())
            .map(|concept| concept.facts.len())
            .sum()
    }

    fn example_count(&self) -> usize {
        self.topics
            .iter()
            .flat_map(|topic| topic.subtopics.iter())
            .flat_map(|subtopic| subtopic.concepts.iter())
            .map(|concept| concept.examples.len())
            .sum()
    }

    fn relation_count(&self) -> usize {
        self.topics
            .iter()
            .flat_map(|topic| topic.subtopics.iter())
            .flat_map(|subtopic| subtopic.concepts.iter())
            .map(|concept| concept.relations.len())
            .sum()
    }

    fn is_real_enough(&self, config: &KnowledgePackConfig) -> bool {
        if !config.require_coverage {
            return self.accepted_count() >= config.min_items_per_domain;
        }
        self.accepted_count() >= config.min_items_per_domain
            && self.topic_count() >= config.min_topics
            && self.subtopic_count() >= config.min_subtopics
            && self.concept_count() >= config.min_concepts
    }

    fn is_tiny_or_legacy_cache(&self, config: &KnowledgePackConfig) -> bool {
        config.reject_tiny_packs
            && (self.topics.is_empty()
                || self.concept_count() < config.min_concepts
                || self.accepted_count() < config.min_items_per_domain)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeTopic {
    #[serde(default)]
    name: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    subtopics: Vec<KnowledgeSubtopic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeSubtopic {
    #[serde(default)]
    name: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    concepts: Vec<KnowledgeConcept>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeConcept {
    #[serde(default)]
    name: String,
    #[serde(default)]
    definition: String,
    #[serde(default)]
    difficulty: String,
    #[serde(default)]
    facts: Vec<String>,
    #[serde(default)]
    examples: Vec<String>,
    #[serde(default)]
    qa_pairs: Vec<KnowledgeQaPair>,
    #[serde(default)]
    relations: Vec<KnowledgeRelation>,
    #[serde(default)]
    source_notes: String,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeQaPair {
    #[serde(default)]
    question: String,
    #[serde(default)]
    answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeRelation {
    #[serde(default)]
    relation_type: String,
    #[serde(default)]
    target: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KnowledgePackItem {
    id: String,
    theme_path: Vec<String>,
    question: String,
    answer: String,
    teacher_confidence: f32,
    validator_score: f32,
    accepted: bool,
    tags: Vec<String>,
    verification_notes: String,
    source_provider: String,
    validator_provider: String,
    created_unix_ms: u64,
    used_count: usize,
    last_used_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BenchmarkKnowledgePackStats {
    pub enabled: bool,
    pub mode: String,
    pub dir: String,
    pub pack_hits: usize,
    pub pack_misses: usize,
    pub pack_items_reused: usize,
    pub pack_items_generated: usize,
    pub pack_items_saved: usize,
    pub pack_rejected_items_saved: usize,
    pub pack_files_loaded: usize,
    pub pack_files_written: usize,
    pub pack_only_starved_jobs: usize,
    pub real_pack_builds: usize,
    pub tiny_packs_rejected: usize,
    pub structured_topics: usize,
    pub structured_subtopics: usize,
    pub structured_concepts: usize,
    pub structured_facts: usize,
    pub structured_examples: usize,
    pub structured_relations: usize,
    pub extracted_training_items: usize,
    pub domains_touched: Vec<String>,
}

impl BenchmarkKnowledgePackStats {
    fn new(config: &KnowledgePackConfig) -> Self {
        Self {
            enabled: config.enabled,
            mode: config.mode.clone(),
            dir: config.dir.clone(),
            ..Self::default()
        }
    }

    fn touch_domain(&mut self, domain: &str) {
        if !self.domains_touched.iter().any(|value| value == domain) {
            self.domains_touched.push(domain.to_string());
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DimensionConfig {
    pub enabled: bool,
    pub dimensions: usize,
    pub cross_validate: bool,
    pub min_agreement: usize,
    pub max_cross_validations: usize,
    pub loss_agreement_margin: f32,
    pub diversity_bonus: f32,
    pub selection: String,
    pub converge_outputs: bool,
    pub output_convergence_radius: usize,
}

impl DimensionConfig {
    fn from_env() -> Self {
        let dimensions = env_usize("BRICKS_AI_PARALLEL_DIMENSIONS").unwrap_or(8).clamp(1, 64);
        Self {
            enabled: env_bool("BRICKS_AI_PARALLEL_DIMENSIONS_ENABLED").unwrap_or(true) && dimensions > 1,
            dimensions,
            cross_validate: env_bool("BRICKS_AI_DIMENSION_CROSS_VALIDATE").unwrap_or(true),
            min_agreement: env_usize("BRICKS_AI_DIMENSION_MIN_AGREEMENT").unwrap_or(2).clamp(1, dimensions),
            max_cross_validations: env_usize("BRICKS_AI_DIMENSION_MAX_CROSS_VALIDATIONS").unwrap_or(dimensions.saturating_sub(1)).clamp(0, dimensions.saturating_sub(1)),
            loss_agreement_margin: env_f32("BRICKS_AI_DIMENSION_LOSS_AGREEMENT_MARGIN").unwrap_or(0.12).max(0.0),
            diversity_bonus: env_f32("BRICKS_AI_DIMENSION_DIVERSITY_BONUS").unwrap_or(0.10).clamp(0.0, 1.0),
            selection: env::var("BRICKS_AI_DIMENSION_SELECTION").unwrap_or_else(|_| "best_validated".to_string()),
            converge_outputs: env_bool("BRICKS_AI_DIMENSION_CONVERGE_OUTPUTS").unwrap_or(true),
            output_convergence_radius: env_usize("BRICKS_AI_DIMENSION_OUTPUT_CONVERGENCE_RADIUS").unwrap_or(4),
        }
    }

    fn active_dimension_count(&self) -> usize {
        if self.enabled { self.dimensions } else { 1 }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ConvergenceConfig {
    pub enabled: bool,
    pub cluster_radius: usize,
    pub min_votes: usize,
    pub require_cross_validation: bool,
    pub reinforce_supporting_paths: bool,
    pub final_boost_repeats: usize,
    pub path_boost_repeats: usize,
    pub correlation_boost_repeats: usize,
    pub correlation_score_boost: f32,
    pub correlation_coefficient_boost: f32,
    pub correlation_survival_min_visits: usize,
    pub correlation_survival_bonus: f32,
    pub weight_votes: f32,
    pub weight_candidate_score: f32,
    pub weight_loss_reward: f32,
    pub weight_stability: f32,
}

impl ConvergenceConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("BRICKS_AI_CONVERGENCE_ENABLED").unwrap_or(true),
            cluster_radius: env_usize("BRICKS_AI_CONVERGENCE_CLUSTER_RADIUS").unwrap_or(4),
            min_votes: env_usize("BRICKS_AI_CONVERGENCE_MIN_VOTES").unwrap_or(2).max(1),
            require_cross_validation: env_bool("BRICKS_AI_CONVERGENCE_REQUIRE_CROSS_VALIDATION").unwrap_or(false),
            reinforce_supporting_paths: env_bool("BRICKS_AI_CONVERGENCE_REINFORCE_SUPPORTING_PATHS").unwrap_or(true),
            final_boost_repeats: env_usize("BRICKS_AI_CONVERGENCE_FINAL_BOOST_REPEATS").unwrap_or(2).max(1),
            path_boost_repeats: env_usize("BRICKS_AI_CONVERGENCE_PATH_BOOST_REPEATS").unwrap_or(1).max(1),
            correlation_boost_repeats: env_usize("BRICKS_AI_CONVERGENCE_CORRELATION_BOOST_REPEATS").unwrap_or(2).clamp(1, 64),
            correlation_score_boost: env_f32("BRICKS_AI_CONVERGENCE_CORRELATION_SCORE_BOOST").unwrap_or(2.2).clamp(0.0, 10.0),
            correlation_coefficient_boost: env_f32("BRICKS_AI_CONVERGENCE_CORRELATION_COEFFICIENT_BOOST").unwrap_or(1.6).clamp(0.0, 10.0),
            correlation_survival_min_visits: env_usize("BRICKS_AI_CONVERGENCE_CORRELATION_SURVIVAL_MIN_VISITS").unwrap_or(3).max(1),
            correlation_survival_bonus: env_f32("BRICKS_AI_CONVERGENCE_CORRELATION_SURVIVAL_BONUS").unwrap_or(0.012).clamp(0.0, 0.10),
            weight_votes: env_f32("BRICKS_AI_CONVERGENCE_WEIGHT_VOTES").unwrap_or(0.45).max(0.0),
            weight_candidate_score: env_f32("BRICKS_AI_CONVERGENCE_WEIGHT_CANDIDATE_SCORE").unwrap_or(0.20).max(0.0),
            weight_loss_reward: env_f32("BRICKS_AI_CONVERGENCE_WEIGHT_LOSS_REWARD").unwrap_or(0.20).max(0.0),
            weight_stability: env_f32("BRICKS_AI_CONVERGENCE_WEIGHT_STABILITY").unwrap_or(0.15).max(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PreFinalDestructionConfig {
    pub enabled: bool,
    pub min_candidate_score: f32,
    pub max_candidate_loss: f32,
    pub min_alignment_score: f32,
    pub protect_case_confidence: f32,
    pub destroy_middle_nodes: bool,
    pub destroy_output_nodes: bool,
    pub always_keep_best_candidate: bool,
}

impl PreFinalDestructionConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("BRICKS_AI_PREFINAL_DESTRUCTION_ENABLED").unwrap_or(true),
            min_candidate_score: env_f32("BRICKS_AI_PREFINAL_MIN_CANDIDATE_SCORE").unwrap_or(0.78).clamp(0.0, 1.0),
            max_candidate_loss: env_f32("BRICKS_AI_PREFINAL_MAX_CANDIDATE_LOSS").unwrap_or(0.35).max(0.0),
            min_alignment_score: env_f32("BRICKS_AI_PREFINAL_MIN_ALIGNMENT_SCORE").unwrap_or(0.78).clamp(0.0, 1.0),
            protect_case_confidence: env_f32("BRICKS_AI_PREFINAL_PROTECT_CASE_CONFIDENCE").unwrap_or(0.18).clamp(0.0, 1.0),
            destroy_middle_nodes: env_bool("BRICKS_AI_PREFINAL_DESTROY_MIDDLE_NODES").unwrap_or(true),
            destroy_output_nodes: env_bool("BRICKS_AI_PREFINAL_DESTROY_OUTPUT_NODES").unwrap_or(true),
            always_keep_best_candidate: env_bool("BRICKS_AI_PREFINAL_ALWAYS_KEEP_BEST_CANDIDATE").unwrap_or(true),
        }
    }
}

#[derive(Debug, Clone)]
struct DimensionPath {
    dimension_id: usize,
    input: NodeId,
    mid_1: NodeId,
    mid_2: NodeId,
    output: NodeId,
}

impl DimensionPath {
    fn nodes(&self) -> [NodeId; 4] {
        [self.input, self.mid_1, self.mid_2, self.output]
    }

    fn key(&self) -> String {
        format!(
            "d{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            self.dimension_id,
            self.input.grid,
            self.input.page,
            self.input.case,
            self.mid_1.grid,
            self.mid_1.page,
            self.mid_1.case,
            self.mid_2.grid,
            self.mid_2.page,
            self.mid_2.case,
            self.output.grid,
            self.output.page,
            self.output.case,
        )
    }
}

#[derive(Debug, Clone)]
struct DimensionCandidateScore {
    path: DimensionPath,
    loss: f32,
    score: f32,
    convergence_score: f32,
    cross_validated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkDimensionEvent {
    pub elapsed_ms: u64,
    pub internal_tick: usize,
    pub theme_path: Vec<String>,
    pub winner_dimension_id: usize,
    pub winner_loss: f32,
    pub winner_score: f32,
    pub agreement_count: usize,
    pub cross_validated_paths: usize,
    pub candidate_count: usize,
    pub winner_path_key: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkDimensionStats {
    pub enabled: bool,
    pub parallel_dimensions: usize,
    pub race_items: usize,
    pub total_dimension_candidates: usize,
    pub winner_counts: Vec<usize>,
    pub unique_winner_paths: usize,
    pub unique_paths_created: usize,
    pub cross_validated_paths: usize,
    pub total_agreement_count: usize,
    pub avg_agreement_count: f32,
    pub avg_winner_loss: f32,
    pub best_winner_loss: f32,
    pub worst_winner_loss: f32,
    pub winner_switch_count: usize,
    pub last_winner_id: Option<usize>,
}

impl BenchmarkDimensionStats {
    fn new(config: &DimensionConfig) -> Self {
        Self {
            enabled: config.enabled,
            parallel_dimensions: config.active_dimension_count(),
            race_items: 0,
            total_dimension_candidates: 0,
            winner_counts: vec![0; config.active_dimension_count()],
            unique_winner_paths: 0,
            unique_paths_created: 0,
            cross_validated_paths: 0,
            total_agreement_count: 0,
            avg_agreement_count: 0.0,
            avg_winner_loss: 0.0,
            best_winner_loss: f32::MAX,
            worst_winner_loss: 0.0,
            winner_switch_count: 0,
            last_winner_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkConvergenceStats {
    pub enabled: bool,
    pub cluster_radius: usize,
    pub min_votes: usize,
    pub candidates_total: usize,
    pub clusters_total: usize,
    pub winner_votes_total: usize,
    pub avg_winner_votes: f32,
    pub avg_winner_score: f32,
    pub max_winner_score: f32,
    pub neighbor_merges: usize,
    pub supporting_paths_reinforced: usize,
    pub singleton_rejections: usize,
    pub last_winner_case: Option<NodeId>,
}

impl BenchmarkConvergenceStats {
    fn new(config: &ConvergenceConfig) -> Self {
        Self {
            enabled: config.enabled,
            cluster_radius: config.cluster_radius,
            min_votes: config.min_votes,
            candidates_total: 0,
            clusters_total: 0,
            winner_votes_total: 0,
            avg_winner_votes: 0.0,
            avg_winner_score: 0.0,
            max_winner_score: 0.0,
            neighbor_merges: 0,
            supporting_paths_reinforced: 0,
            singleton_rejections: 0,
            last_winner_case: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BenchmarkPreFinalDestructionStats {
    pub enabled: bool,
    pub runs: usize,
    pub candidates_seen: usize,
    pub candidates_forwarded: usize,
    pub candidates_destroyed: usize,
    pub candidates_rescued: usize,
    pub cases_destroyed: usize,
    pub correlations_destroyed: usize,
    pub blocks_destroyed: usize,
    pub last_alignment_score: f32,
    pub last_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct PreFinalDestructionRunStats {
    candidates_seen: usize,
    candidates_forwarded: usize,
    candidates_destroyed: usize,
    candidates_rescued: usize,
    cases_destroyed: usize,
    correlations_destroyed: usize,
    blocks_destroyed: usize,
    last_alignment_score: f32,
    last_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkSleepInterruptionEvent {
    pub detected_at_elapsed_ms: u64,
    pub previous_elapsed_ms: u64,
    pub gap_ms: u64,
    pub charged_sleep_ms: u64,
    pub internal_tick: usize,
    pub kind: String,
}

#[derive(Debug, Clone)]
struct DimensionTrainingResult {
    winner_dimension_id: usize,
    winner_loss: f32,
    winner_score: f32,
    agreement_count: usize,
    cross_validated_paths: usize,
    candidate_count: usize,
    pre_final_stats: PreFinalDestructionRunStats,
    winner_path_key: String,
    created_paths: usize,
    convergence_score: f32,
    convergence_votes: usize,
    convergence_final_node: NodeId,
    convergence_neighbor_merges: usize,
    convergence_supporting_paths_reinforced: usize,
    convergence_singleton_rejections: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkProviderInfo {
    pub name: String,
    pub model: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkHardwareSnapshot {
    pub unix_ms: u64,
    pub cpu_name: Option<String>,
    pub cpu_logical_cores: usize,
    pub memory_total_kb: Option<u64>,
    pub memory_free_kb: Option<u64>,
    pub gpu_name: Option<String>,
    pub gpu_memory_total_mb: Option<u64>,
    pub gpu_memory_used_mb: Option<u64>,
    pub gpu_utilization_percent: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct BenchmarkTimingTotals {
    pub subtheme_expansion_ms: u64,
    pub training_data_generation_ms: u64,
    pub knowledge_pack_load_ms: u64,
    pub knowledge_pack_save_ms: u64,
    pub validation_ms: u64,
    pub model_training_ms: u64,
    pub checkpoint_save_ms: u64,
    pub pause_ms: u64,
    pub sleep_interruption_ms: u64,
    pub longest_sleep_interruption_ms: u64,
    pub interruption_count: usize,
    pub convergence_ms: u64,
    pub wall_elapsed_ms: u64,
    pub active_elapsed_ms: u64,
    pub avg_wall_ms_per_new_trained_item: f32,
    pub avg_active_ms_per_new_trained_item: f32,
    pub avg_ms_per_internal_tick: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkModelMetrics {
    pub engraved_cases: usize,
    pub engraved_correlations: usize,
    pub active_correlations: usize,
    pub preferred_paths: usize,
    pub avg_case_confidence: f32,
    pub max_case_confidence: f32,
    pub avg_correlation_score: f32,
    pub max_correlation_score: f32,
    pub model_size_bytes: Option<u64>,
    pub checkpoint_size_bytes: Option<u64>,
    pub final_probe_prediction: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkStateSnapshot {
    pub trained_items: usize,
    pub accepted_items: usize,
    pub rejected_items: usize,
    pub queue_remaining: usize,
    pub pack_status: String,
    pub model: BenchmarkModelMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkEvent {
    pub elapsed_ms: u64,
    pub duration_ms: Option<u64>,
    pub internal_tick: usize,
    pub kind: String,
    pub provider: Option<String>,
    pub theme_path: Vec<String>,
    pub generated_items: usize,
    pub accepted_delta: usize,
    pub rejected_delta: usize,
    pub trained_delta: usize,
    pub state: BenchmarkStateSnapshot,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkFinalSummary {
    pub elapsed_ms: u64,
    pub active_elapsed_ms: u64,
    pub pause_ms: u64,
    pub sleep_interruption_ms: u64,
    pub interruption_count: usize,
    pub longest_sleep_interruption_ms: u64,
    pub internal_ticks: usize,
    pub requested_steps: usize,
    pub target_trained_items: usize,
    pub reached_target: bool,
    pub stopped_by_user: bool,
    pub cutoff_reason: String,
    pub new_trained_items: usize,
    pub accepted_items: usize,
    pub rejected_items: usize,
    pub trained_items: usize,
    pub acceptance_rate: f32,
    pub queue_remaining: usize,
    pub avg_wall_ms_per_new_trained_item: f32,
    pub avg_active_ms_per_new_trained_item: f32,
    pub avg_ms_per_internal_tick: f32,
    pub final_model: BenchmarkModelMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub format: String,
    pub version: u32,
    pub command_line: String,
    pub requested_steps: usize,
    pub items_per_theme: usize,
    pub max_depth: usize,
    pub checkpoint_path: String,
    pub provider_mode: String,
    pub providers: Vec<BenchmarkProviderInfo>,
    pub device: ComputeInfo,
    pub dimension_settings: DimensionConfig,
    pub convergence_settings: ConvergenceConfig,
    pub pre_final_destruction_settings: PreFinalDestructionConfig,
    pub knowledge_pack_settings: KnowledgePackConfig,
    pub knowledge_pack_stats: BenchmarkKnowledgePackStats,
    pub dimension_stats: BenchmarkDimensionStats,
    pub convergence_stats: BenchmarkConvergenceStats,
    pub pre_final_destruction_stats: BenchmarkPreFinalDestructionStats,
    pub dimension_events: Vec<BenchmarkDimensionEvent>,
    pub sleep_interruption_events: Vec<BenchmarkSleepInterruptionEvent>,
    pub timing: BenchmarkTimingTotals,
    pub start_unix_ms: u64,
    pub start_hardware: BenchmarkHardwareSnapshot,
    pub end_hardware: Option<BenchmarkHardwareSnapshot>,
    pub initial_state: BenchmarkStateSnapshot,
    pub events: Vec<BenchmarkEvent>,
    pub final_summary: Option<BenchmarkFinalSummary>,
}

struct BenchmarkRecorder {
    start: Instant,
    report: BenchmarkReport,
    dimension_paths_seen: HashSet<String>,
    last_progress_elapsed_ms: u64,
    sleep_gap_threshold_ms: u64,
}

impl BenchmarkRecorder {
    fn new(
        steps: usize,
        items_per_theme: usize,
        depth: usize,
        checkpoint_path: &str,
        device: &ComputeInfo,
        pool: &TeacherProviderPool,
        trainer: &RawAITrainer,
        dimension_settings: DimensionConfig,
        convergence_settings: ConvergenceConfig,
        pre_final_destruction_settings: PreFinalDestructionConfig,
        knowledge_pack_settings: KnowledgePackConfig,
    ) -> Self {
        let start_hardware = capture_hardware_snapshot(device);
        let providers = pool
            .providers
            .iter()
            .map(|provider| BenchmarkProviderInfo {
                name: provider.display_name().to_string(),
                model: provider.model.clone(),
                role: format!("{:?}", provider.provider),
            })
            .collect();

        let dimension_stats = BenchmarkDimensionStats::new(&dimension_settings);
        let convergence_stats = BenchmarkConvergenceStats::new(&convergence_settings);
        let pre_final_destruction_stats = BenchmarkPreFinalDestructionStats {
            enabled: pre_final_destruction_settings.enabled,
            ..BenchmarkPreFinalDestructionStats::default()
        };
        let knowledge_pack_stats = BenchmarkKnowledgePackStats::new(&knowledge_pack_settings);

        Self {
            start: Instant::now(),
            dimension_paths_seen: HashSet::new(),
            last_progress_elapsed_ms: 0,
            sleep_gap_threshold_ms: env_usize("BRICKS_AI_SLEEP_GAP_THRESHOLD_MS").unwrap_or(300_000) as u64,
            report: BenchmarkReport {
                format: "bricks-ai-benchmark".to_string(),
                version: 6,
                command_line: env::args().collect::<Vec<_>>().join(" "),
                requested_steps: steps,
                items_per_theme,
                max_depth: depth,
                checkpoint_path: checkpoint_path.to_string(),
                provider_mode: format!("{:?}", pool.mode),
                providers,
                device: device.clone(),
                dimension_settings,
                convergence_settings,
                pre_final_destruction_settings,
                knowledge_pack_settings,
                knowledge_pack_stats,
                dimension_stats,
                convergence_stats,
                pre_final_destruction_stats,
                dimension_events: Vec::new(),
                sleep_interruption_events: Vec::new(),
                timing: BenchmarkTimingTotals::default(),
                start_unix_ms: unix_ms_now(),
                start_hardware,
                end_hardware: None,
                initial_state: benchmark_state_snapshot(trainer, None, None, None),
                events: Vec::new(),
                final_summary: None,
            },
        }
    }

    fn elapsed_ms(&self) -> u64 {
        millis_u64(self.start.elapsed())
    }

    fn add_timing(&mut self, kind: &str, duration: Duration) {
        let ms = millis_u64(duration);
        match kind {
            "subtheme_expansion" => self.report.timing.subtheme_expansion_ms = self.report.timing.subtheme_expansion_ms.saturating_add(ms),
            "training_data_generation" => self.report.timing.training_data_generation_ms = self.report.timing.training_data_generation_ms.saturating_add(ms),
            "knowledge_pack_load" => self.report.timing.knowledge_pack_load_ms = self.report.timing.knowledge_pack_load_ms.saturating_add(ms),
            "knowledge_pack_save" => self.report.timing.knowledge_pack_save_ms = self.report.timing.knowledge_pack_save_ms.saturating_add(ms),
            "validation" => self.report.timing.validation_ms = self.report.timing.validation_ms.saturating_add(ms),
            "model_training" => self.report.timing.model_training_ms = self.report.timing.model_training_ms.saturating_add(ms),
            "checkpoint_save" => self.report.timing.checkpoint_save_ms = self.report.timing.checkpoint_save_ms.saturating_add(ms),
            "convergence" => self.report.timing.convergence_ms = self.report.timing.convergence_ms.saturating_add(ms),
            _ => {}
        }
    }

    fn detect_sleep_interruption(&mut self, internal_tick: usize, kind: &str) {
        let now = self.elapsed_ms();
        if self.last_progress_elapsed_ms == 0 {
            self.last_progress_elapsed_ms = now;
            return;
        }
        let gap = now.saturating_sub(self.last_progress_elapsed_ms);
        if gap > self.sleep_gap_threshold_ms {
            let charged = gap.saturating_sub(self.sleep_gap_threshold_ms);
            self.report.timing.sleep_interruption_ms = self.report.timing.sleep_interruption_ms.saturating_add(charged);
            self.report.timing.longest_sleep_interruption_ms = self.report.timing.longest_sleep_interruption_ms.max(charged);
            self.report.timing.interruption_count += 1;
            self.report.sleep_interruption_events.push(BenchmarkSleepInterruptionEvent {
                detected_at_elapsed_ms: now,
                previous_elapsed_ms: self.last_progress_elapsed_ms,
                gap_ms: gap,
                charged_sleep_ms: charged,
                internal_tick,
                kind: kind.to_string(),
            });
        }
        self.last_progress_elapsed_ms = now;
    }

    fn record_event(
        &mut self,
        trainer: &RawAITrainer,
        internal_tick: usize,
        kind: &str,
        provider: Option<&str>,
        theme_path: &[String],
        generated_items: usize,
        accepted_delta: usize,
        rejected_delta: usize,
        trained_delta: usize,
    ) {
        self.record_event_with_duration(
            trainer,
            internal_tick,
            kind,
            provider,
            theme_path,
            generated_items,
            accepted_delta,
            rejected_delta,
            trained_delta,
            None,
        );
    }

    fn record_event_with_duration(
        &mut self,
        trainer: &RawAITrainer,
        internal_tick: usize,
        kind: &str,
        provider: Option<&str>,
        theme_path: &[String],
        generated_items: usize,
        accepted_delta: usize,
        rejected_delta: usize,
        trained_delta: usize,
        duration: Option<Duration>,
    ) {
        self.detect_sleep_interruption(internal_tick, kind);
        self.report.events.push(BenchmarkEvent {
            elapsed_ms: self.elapsed_ms(),
            duration_ms: duration.map(millis_u64),
            internal_tick,
            kind: kind.to_string(),
            provider: provider.map(str::to_string),
            theme_path: theme_path.to_vec(),
            generated_items,
            accepted_delta,
            rejected_delta,
            trained_delta,
            state: benchmark_state_snapshot(trainer, None, None, None),
        });
    }

    fn record_dimension_result(
        &mut self,
        internal_tick: usize,
        theme_path: &[String],
        result: &DimensionTrainingResult,
    ) {
        self.detect_sleep_interruption(internal_tick, "dimension_race");
        let stats = &mut self.report.dimension_stats;
        stats.race_items += 1;
        stats.total_dimension_candidates += result.candidate_count;
        if result.winner_dimension_id >= stats.winner_counts.len() {
            stats.winner_counts.resize(result.winner_dimension_id + 1, 0);
        }
        stats.winner_counts[result.winner_dimension_id] += 1;
        stats.total_agreement_count += result.agreement_count;
        stats.avg_agreement_count = stats.total_agreement_count as f32 / stats.race_items.max(1) as f32;
        stats.cross_validated_paths += result.cross_validated_paths;
        stats.unique_paths_created += result.created_paths;

        if self.dimension_paths_seen.insert(result.winner_path_key.clone()) {
            stats.unique_winner_paths = self.dimension_paths_seen.len();
        }

        let race_count = stats.race_items as f32;
        stats.avg_winner_loss = ((stats.avg_winner_loss * (race_count - 1.0)) + result.winner_loss) / race_count;
        stats.best_winner_loss = stats.best_winner_loss.min(result.winner_loss);
        stats.worst_winner_loss = stats.worst_winner_loss.max(result.winner_loss);

        if let Some(last) = stats.last_winner_id {
            if last != result.winner_dimension_id {
                stats.winner_switch_count += 1;
            }
        }
        stats.last_winner_id = Some(result.winner_dimension_id);

        let conv = &mut self.report.convergence_stats;
        conv.candidates_total += result.candidate_count;
        conv.clusters_total += 1;
        conv.winner_votes_total += result.convergence_votes;
        conv.avg_winner_votes = conv.winner_votes_total as f32 / conv.clusters_total.max(1) as f32;
        conv.avg_winner_score = ((conv.avg_winner_score * ((conv.clusters_total.saturating_sub(1)) as f32)) + result.convergence_score) / conv.clusters_total.max(1) as f32;
        conv.max_winner_score = conv.max_winner_score.max(result.convergence_score);
        conv.neighbor_merges += result.convergence_neighbor_merges;
        conv.supporting_paths_reinforced += result.convergence_supporting_paths_reinforced;
        conv.singleton_rejections += result.convergence_singleton_rejections;
        conv.last_winner_case = Some(result.convergence_final_node);

        self.report.dimension_events.push(BenchmarkDimensionEvent {
            elapsed_ms: self.elapsed_ms(),
            internal_tick,
            theme_path: theme_path.to_vec(),
            winner_dimension_id: result.winner_dimension_id,
            winner_loss: result.winner_loss,
            winner_score: result.winner_score,
            agreement_count: result.agreement_count,
            cross_validated_paths: result.cross_validated_paths,
            candidate_count: result.candidate_count,
            winner_path_key: result.winner_path_key.clone(),
        });
    }

    fn record_pre_final_destruction(&mut self, run: &PreFinalDestructionRunStats) {
        let stats = &mut self.report.pre_final_destruction_stats;
        stats.runs += 1;
        stats.candidates_seen += run.candidates_seen;
        stats.candidates_forwarded += run.candidates_forwarded;
        stats.candidates_destroyed += run.candidates_destroyed;
        stats.candidates_rescued += run.candidates_rescued;
        stats.cases_destroyed += run.cases_destroyed;
        stats.correlations_destroyed += run.correlations_destroyed;
        stats.blocks_destroyed += run.blocks_destroyed;
        stats.last_alignment_score = run.last_alignment_score;
        if let Some(reason) = &run.last_reason {
            stats.last_reason = Some(reason.clone());
        }
    }

    fn record_knowledge_pack_load(&mut self, domain: &str, hit_count: usize, loaded_file: bool) {
        let stats = &mut self.report.knowledge_pack_stats;
        stats.touch_domain(domain);
        if loaded_file {
            stats.pack_files_loaded += 1;
        }
        if hit_count > 0 {
            stats.pack_hits += 1;
            stats.pack_items_reused += hit_count;
        } else {
            stats.pack_misses += 1;
        }
    }

    fn record_knowledge_pack_saved(&mut self, domain: &str, accepted_saved: usize, rejected_saved: usize, wrote_file: bool) {
        let stats = &mut self.report.knowledge_pack_stats;
        stats.touch_domain(domain);
        stats.pack_items_saved += accepted_saved;
        stats.pack_rejected_items_saved += rejected_saved;
        stats.pack_items_generated += accepted_saved + rejected_saved;
        if wrote_file {
            stats.pack_files_written += 1;
        }
    }

    fn record_knowledge_pack_starved(&mut self, domain: &str) {
        let stats = &mut self.report.knowledge_pack_stats;
        stats.touch_domain(domain);
        stats.pack_only_starved_jobs += 1;
    }

    fn record_real_pack_build(&mut self, domain: &str, pack: &KnowledgePack, extracted_items: usize, tiny_rejected: bool) {
        let stats = &mut self.report.knowledge_pack_stats;
        stats.touch_domain(domain);
        stats.real_pack_builds += 1;
        if tiny_rejected {
            stats.tiny_packs_rejected += 1;
        }
        stats.structured_topics += pack.topic_count();
        stats.structured_subtopics += pack.subtopic_count();
        stats.structured_concepts += pack.concept_count();
        stats.structured_facts += pack.fact_count();
        stats.structured_examples += pack.example_count();
        stats.structured_relations += pack.relation_count();
        stats.extracted_training_items += extracted_items;
    }

    fn finish(
        &mut self,
        trainer: &RawAITrainer,
        device: &ComputeInfo,
        internal_ticks: usize,
        target_trained_items: usize,
        final_probe_prediction: f32,
        model_path: &str,
        checkpoint_path: &str,
        pause_ms: u64,
        stopped_by_user: bool,
        cutoff_reason: &str,
    ) {
        self.report.end_hardware = Some(capture_hardware_snapshot(device));
        let final_model = benchmark_model_metrics(
            trainer,
            &trainer.export_engraved_model(),
            Some(model_path),
            Some(checkpoint_path),
            Some(final_probe_prediction),
        );
        let trained_items = trainer.pack_state.trained_items;
        let accepted_items = trainer.pack_state.accepted_items;
        let rejected_items = trainer.pack_state.rejected_items;
        let reviewed_items = accepted_items + rejected_items;
        let acceptance_rate = if reviewed_items == 0 {
            0.0
        } else {
            accepted_items as f32 / reviewed_items as f32
        };
        let elapsed_ms = self.elapsed_ms();
        let sleep_ms = self.report.timing.sleep_interruption_ms;
        let active_elapsed_ms = elapsed_ms.saturating_sub(pause_ms).saturating_sub(sleep_ms);
        let initial_trained = self.report.initial_state.trained_items;
        let new_trained_items = trained_items.saturating_sub(initial_trained);
        let avg_wall = average_ms_per_item(elapsed_ms, new_trained_items);
        let avg_active = average_ms_per_item(active_elapsed_ms, new_trained_items);
        let avg_tick = if internal_ticks == 0 { 0.0 } else { active_elapsed_ms as f32 / internal_ticks as f32 };

        self.report.timing.pause_ms = pause_ms;
        self.report.timing.sleep_interruption_ms = sleep_ms;
        self.report.timing.wall_elapsed_ms = elapsed_ms;
        self.report.timing.active_elapsed_ms = active_elapsed_ms;
        self.report.timing.avg_wall_ms_per_new_trained_item = avg_wall;
        self.report.timing.avg_active_ms_per_new_trained_item = avg_active;
        self.report.timing.avg_ms_per_internal_tick = avg_tick;

        self.report.final_summary = Some(BenchmarkFinalSummary {
            elapsed_ms,
            active_elapsed_ms,
            pause_ms,
            sleep_interruption_ms: sleep_ms,
            interruption_count: self.report.timing.interruption_count,
            longest_sleep_interruption_ms: self.report.timing.longest_sleep_interruption_ms,
            internal_ticks,
            requested_steps: self.report.requested_steps,
            target_trained_items,
            reached_target: trained_items >= target_trained_items,
            stopped_by_user,
            cutoff_reason: cutoff_reason.to_string(),
            new_trained_items,
            accepted_items,
            rejected_items,
            trained_items,
            acceptance_rate,
            queue_remaining: trainer.pack_state.queue.len(),
            avg_wall_ms_per_new_trained_item: avg_wall,
            avg_active_ms_per_new_trained_item: avg_active,
            avg_ms_per_internal_tick: avg_tick,
            final_model,
        });
    }
}


fn run_train_pack_entry(args: &[String]) -> Result<(), Box<dyn Error>> {
    run_teacher_pack_training_local(args)
}

fn run_device_diagnostics() -> Result<(), Box<dyn Error>> {
    let device = detect_local_compute();
    print_compute_summary(&device);
    Ok(())
}

fn detect_local_compute() -> ComputeInfo {
    let preference = match env::var("BRICKS_AI_DEVICE")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "cpu" => ComputePreference::Cpu,
        "gpu" => ComputePreference::Gpu,
        _ => ComputePreference::Auto,
    };

    let gpu_name = detect_supported_gpu();
    match preference {
        ComputePreference::Cpu => ComputeInfo {
            preference,
            effective_gpu: false,
            gpu_name,
            reason: "CPU forced by BRICKS_AI_DEVICE=cpu".to_string(),
        },
        ComputePreference::Gpu | ComputePreference::Auto => {
            if let Some(name) = gpu_name.clone() {
                ComputeInfo {
                    preference,
                    effective_gpu: true,
                    gpu_name: Some(name.clone()),
                    reason: format!("compatible local GPU detected: {name}"),
                }
            } else {
                let reason = if preference == ComputePreference::Gpu {
                    "GPU requested but no compatible local GPU was detected; falling back to CPU".to_string()
                } else {
                    "no compatible local GPU detected; using CPU".to_string()
                };
                ComputeInfo {
                    preference,
                    effective_gpu: false,
                    gpu_name: None,
                    reason,
                }
            }
        }
    }
}

fn detect_supported_gpu() -> Option<String> {
    detect_gpu_with_nvidia_smi().or_else(detect_gpu_with_wmic)
}

fn detect_gpu_with_nvidia_smi() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().map(str::trim).find(|line| !line.is_empty()).map(str::to_string)
}

fn detect_gpu_with_wmic() -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    let output = Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "Name"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.eq_ignore_ascii_case("name"))
        .find(|line| {
            let upper = line.to_ascii_uppercase();
            upper.contains("NVIDIA")
                || upper.contains("GEFORCE")
                || upper.contains("RTX")
                || upper.contains("GTX")
                || upper.contains("QUADRO")
                || upper.contains("TESLA")
                || upper.contains("RADEON")
                || upper.contains("INTEL ARC")
        })
        .map(str::to_string)
}

fn print_compute_summary(info: &ComputeInfo) {
    println!("Bricks AI local compute");
    println!("device_preference={:?}", info.preference);
    println!("effective_device={}", if info.effective_gpu { "GPU" } else { "CPU" });
    if let Some(name) = &info.gpu_name {
        println!("gpu_name={name}");
    }
    println!("reason={}", info.reason);
    if info.effective_gpu {
        println!("note=Ollama and supported local runtimes may use the local GPU automatically.");
    } else {
        println!("note=No compatible local GPU was detected, so Bricks AI uses CPU locally.");
    }
}


fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(millis_u64)
        .unwrap_or(0)
}

fn millis_u64(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn capture_hardware_snapshot(device: &ComputeInfo) -> BenchmarkHardwareSnapshot {
    let (memory_total_kb, memory_free_kb) = detect_system_memory_kb();
    let gpu_metrics = detect_nvidia_gpu_metrics();

    BenchmarkHardwareSnapshot {
        unix_ms: unix_ms_now(),
        cpu_name: detect_cpu_name(),
        cpu_logical_cores: std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1),
        memory_total_kb,
        memory_free_kb,
        gpu_name: gpu_metrics
            .as_ref()
            .and_then(|metrics| metrics.name.clone())
            .or_else(|| device.gpu_name.clone()),
        gpu_memory_total_mb: gpu_metrics.as_ref().and_then(|metrics| metrics.memory_total_mb),
        gpu_memory_used_mb: gpu_metrics.as_ref().and_then(|metrics| metrics.memory_used_mb),
        gpu_utilization_percent: gpu_metrics.as_ref().and_then(|metrics| metrics.utilization_percent),
    }
}

struct NvidiaGpuMetrics {
    name: Option<String>,
    memory_total_mb: Option<u64>,
    memory_used_mb: Option<u64>,
    utilization_percent: Option<u64>,
}

fn detect_nvidia_gpu_metrics() -> Option<NvidiaGpuMetrics> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().map(str::trim).find(|line| !line.is_empty())?;
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next().filter(|value| !value.is_empty()).map(str::to_string);
    let memory_total_mb = parts.next().and_then(|value| value.parse::<u64>().ok());
    let memory_used_mb = parts.next().and_then(|value| value.parse::<u64>().ok());
    let utilization_percent = parts.next().and_then(|value| value.parse::<u64>().ok());

    Some(NvidiaGpuMetrics {
        name,
        memory_total_mb,
        memory_used_mb,
        utilization_percent,
    })
}

fn detect_cpu_name() -> Option<String> {
    if cfg!(windows) {
        let output = Command::new("wmic")
            .args(["cpu", "get", "Name", "/value"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        return text
            .lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix("Name=").map(str::trim).map(str::to_string))
            .filter(|value| !value.is_empty());
    }

    fs::read_to_string("/proc/cpuinfo")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("model name").and_then(|value| value.split_once(':').map(|(_, name)| name.trim().to_string())))
        .filter(|value| !value.is_empty())
}

fn detect_system_memory_kb() -> (Option<u64>, Option<u64>) {
    if cfg!(windows) {
        return detect_windows_memory_kb();
    }
    detect_proc_meminfo_kb()
}

fn detect_windows_memory_kb() -> (Option<u64>, Option<u64>) {
    let output = Command::new("wmic")
        .args(["OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/Value"])
        .output();
    let Ok(output) = output else {
        return (None, None);
    };
    if !output.status.success() {
        return (None, None);
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut total = None;
    let mut free = None;
    for line in text.lines().map(str::trim) {
        if let Some(value) = line.strip_prefix("TotalVisibleMemorySize=") {
            total = value.parse::<u64>().ok();
        } else if let Some(value) = line.strip_prefix("FreePhysicalMemory=") {
            free = value.parse::<u64>().ok();
        }
    }
    (total, free)
}

fn detect_proc_meminfo_kb() -> (Option<u64>, Option<u64>) {
    let Ok(text) = fs::read_to_string("/proc/meminfo") else {
        return (None, None);
    };
    let mut total = None;
    let mut available = None;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            total = line.split_whitespace().nth(1).and_then(|value| value.parse::<u64>().ok());
        } else if line.starts_with("MemAvailable:") {
            available = line.split_whitespace().nth(1).and_then(|value| value.parse::<u64>().ok());
        }
    }
    (total, available)
}

fn benchmark_model_metrics(
    trainer: &RawAITrainer,
    model: &EngravedModel,
    model_path: Option<&str>,
    checkpoint_path: Option<&str>,
    final_probe_prediction: Option<f32>,
) -> BenchmarkModelMetrics {
    let case_count = model.cases.len();
    let correlation_count = model.correlations.len();
    let avg_case_confidence = average_f32(model.cases.iter().map(|case| case.confidence));
    let max_case_confidence = max_f32(model.cases.iter().map(|case| case.confidence));
    let avg_correlation_score = average_f32(model.correlations.iter().map(|correlation| correlation.score));
    let max_correlation_score = max_f32(model.correlations.iter().map(|correlation| correlation.score));

    BenchmarkModelMetrics {
        engraved_cases: case_count,
        engraved_correlations: correlation_count,
        active_correlations: trainer.correlations.iter().filter(|correlation| correlation.active).count(),
        preferred_paths: trainer.paths.iter().filter(|path| path.active).count(),
        avg_case_confidence,
        max_case_confidence,
        avg_correlation_score,
        max_correlation_score,
        model_size_bytes: model_path.and_then(file_size_bytes),
        checkpoint_size_bytes: checkpoint_path.and_then(file_size_bytes),
        final_probe_prediction,
    }
}

fn benchmark_state_snapshot(
    trainer: &RawAITrainer,
    model_path: Option<&str>,
    checkpoint_path: Option<&str>,
    final_probe_prediction: Option<f32>,
) -> BenchmarkStateSnapshot {
    BenchmarkStateSnapshot {
        trained_items: trainer.pack_state.trained_items,
        accepted_items: trainer.pack_state.accepted_items,
        rejected_items: trainer.pack_state.rejected_items,
        queue_remaining: trainer.pack_state.queue.len(),
        pack_status: format!("{:?}", trainer.pack_state.status),
        model: benchmark_model_metrics(
            trainer,
            &trainer.export_engraved_model(),
            model_path,
            checkpoint_path,
            final_probe_prediction,
        ),
    }
}

fn average_f32(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0f32;
    for value in values {
        count += 1;
        sum += value;
    }
    if count == 0 { 0.0 } else { sum / count as f32 }
}

fn max_f32(values: impl Iterator<Item = f32>) -> f32 {
    values.fold(0.0f32, f32::max)
}

fn file_size_bytes(path: &str) -> Option<u64> {
    fs::metadata(path).ok().map(|metadata| metadata.len())
}

fn write_benchmark_files(report: &BenchmarkReport) -> Result<(String, String), Box<dyn Error>> {
    let json_path = env::var("BRICKS_AI_BENCHMARK_JSON").unwrap_or_else(|_| "benchmark_report.json".to_string());
    let csv_path = env::var("BRICKS_AI_BENCHMARK_CSV").unwrap_or_else(|_| "benchmark_report.csv".to_string());
    fs::write(&json_path, serde_json::to_string_pretty(report)?)?;
    fs::write(&csv_path, benchmark_summary_csv(report))?;
    Ok((json_path, csv_path))
}

fn benchmark_summary_csv(report: &BenchmarkReport) -> String {
    let mut csv = String::new();
    let headers = [
        "format", "version", "requested_steps", "items_per_theme", "max_depth", "provider_mode",
        "effective_gpu", "gpu_name", "cpu_logical_cores", "memory_total_kb", "memory_free_kb",
        "elapsed_ms", "active_elapsed_ms", "pause_ms", "sleep_interruption_ms", "interruption_count", "longest_sleep_interruption_ms", "internal_ticks", "new_trained_items",
        "trained_items", "accepted_items", "rejected_items", "acceptance_rate", "queue_remaining",
        "engraved_cases", "engraved_correlations", "active_correlations", "preferred_paths",
        "avg_case_confidence", "max_case_confidence", "avg_correlation_score", "max_correlation_score",
        "model_size_bytes", "checkpoint_size_bytes", "final_probe_prediction", "reached_target",
        "stopped_by_user", "cutoff_reason", "parallel_dimensions", "race_items", "unique_winner_paths",
        "cross_validated_paths", "avg_agreement_count", "avg_winner_loss", "best_winner_loss",
        "worst_winner_loss", "convergence_candidates_total", "convergence_clusters_total", "convergence_winner_votes_total", "convergence_avg_winner_score", "convergence_max_winner_score", "convergence_neighbor_merges", "convergence_supporting_paths_reinforced", "convergence_last_winner_case", "prefinal_runs", "prefinal_candidates_seen", "prefinal_candidates_forwarded", "prefinal_candidates_destroyed", "prefinal_candidates_rescued", "prefinal_cases_destroyed", "prefinal_correlations_destroyed", "prefinal_blocks_destroyed", "prefinal_last_alignment_score", "prefinal_last_reason", "avg_wall_ms_per_new_trained_item", "avg_active_ms_per_new_trained_item",
        "avg_ms_per_internal_tick", "subtheme_expansion_ms", "training_data_generation_ms",
        "validation_ms", "model_training_ms", "convergence_ms", "checkpoint_save_ms", "knowledge_pack_enabled",
        "knowledge_pack_mode", "knowledge_pack_dir", "pack_hits", "pack_misses", "pack_items_reused",
        "pack_items_generated", "pack_items_saved", "pack_rejected_items_saved", "pack_files_loaded",
        "pack_files_written", "pack_only_starved_jobs", "real_pack_builds", "tiny_packs_rejected", "structured_topics", "structured_subtopics", "structured_concepts", "structured_facts", "structured_examples", "structured_relations", "extracted_training_items", "pack_domains_touched", "knowledge_pack_load_ms",
        "knowledge_pack_save_ms",
    ];
    csv.push_str(&headers.join(","));
    csv.push('\n');

    if let Some(summary) = &report.final_summary {
        let hardware = report.end_hardware.as_ref().unwrap_or(&report.start_hardware);
        let model = &summary.final_model;
        let dims = &report.dimension_stats;
        let conv = &report.convergence_stats;
        let packs = &report.knowledge_pack_stats;
        let pre = &report.pre_final_destruction_stats;
        let timing = &report.timing;
        let fields = vec![
            csv_escape(&report.format),
            report.version.to_string(),
            report.requested_steps.to_string(),
            report.items_per_theme.to_string(),
            report.max_depth.to_string(),
            csv_escape(&report.provider_mode),
            report.device.effective_gpu.to_string(),
            csv_escape(hardware.gpu_name.as_deref().unwrap_or("")),
            hardware.cpu_logical_cores.to_string(),
            optional_u64_csv(hardware.memory_total_kb),
            optional_u64_csv(hardware.memory_free_kb),
            summary.elapsed_ms.to_string(),
            summary.active_elapsed_ms.to_string(),
            summary.pause_ms.to_string(),
            summary.sleep_interruption_ms.to_string(),
            summary.interruption_count.to_string(),
            summary.longest_sleep_interruption_ms.to_string(),
            summary.internal_ticks.to_string(),
            summary.new_trained_items.to_string(),
            summary.trained_items.to_string(),
            summary.accepted_items.to_string(),
            summary.rejected_items.to_string(),
            format!("{:.6}", summary.acceptance_rate),
            summary.queue_remaining.to_string(),
            model.engraved_cases.to_string(),
            model.engraved_correlations.to_string(),
            model.active_correlations.to_string(),
            model.preferred_paths.to_string(),
            format!("{:.6}", model.avg_case_confidence),
            format!("{:.6}", model.max_case_confidence),
            format!("{:.6}", model.avg_correlation_score),
            format!("{:.6}", model.max_correlation_score),
            optional_u64_csv(model.model_size_bytes),
            optional_u64_csv(model.checkpoint_size_bytes),
            format!("{:.6}", model.final_probe_prediction.unwrap_or(0.0)),
            summary.reached_target.to_string(),
            summary.stopped_by_user.to_string(),
            csv_escape(&summary.cutoff_reason),
            dims.parallel_dimensions.to_string(),
            dims.race_items.to_string(),
            dims.unique_winner_paths.to_string(),
            dims.cross_validated_paths.to_string(),
            format!("{:.3}", dims.avg_agreement_count),
            format!("{:.6}", dims.avg_winner_loss),
            format!("{:.6}", if dims.best_winner_loss == f32::MAX { 0.0 } else { dims.best_winner_loss }),
            format!("{:.6}", dims.worst_winner_loss),
            conv.candidates_total.to_string(),
            conv.clusters_total.to_string(),
            conv.winner_votes_total.to_string(),
            format!("{:.6}", conv.avg_winner_score),
            format!("{:.6}", conv.max_winner_score),
            conv.neighbor_merges.to_string(),
            conv.supporting_paths_reinforced.to_string(),
            csv_escape(&conv.last_winner_case.map(|node| format!("{}:{}:{}", node.grid, node.page, node.case)).unwrap_or_default()),
            pre.runs.to_string(),
            pre.candidates_seen.to_string(),
            pre.candidates_forwarded.to_string(),
            pre.candidates_destroyed.to_string(),
            pre.candidates_rescued.to_string(),
            pre.cases_destroyed.to_string(),
            pre.correlations_destroyed.to_string(),
            pre.blocks_destroyed.to_string(),
            format!("{:.6}", pre.last_alignment_score),
            csv_escape(pre.last_reason.as_deref().unwrap_or("")),
            format!("{:.3}", summary.avg_wall_ms_per_new_trained_item),
            format!("{:.3}", summary.avg_active_ms_per_new_trained_item),
            format!("{:.3}", summary.avg_ms_per_internal_tick),
            timing.subtheme_expansion_ms.to_string(),
            timing.training_data_generation_ms.to_string(),
            timing.validation_ms.to_string(),
            timing.model_training_ms.to_string(),
            timing.convergence_ms.to_string(),
            timing.checkpoint_save_ms.to_string(),
            packs.enabled.to_string(),
            csv_escape(&packs.mode),
            csv_escape(&packs.dir),
            packs.pack_hits.to_string(),
            packs.pack_misses.to_string(),
            packs.pack_items_reused.to_string(),
            packs.pack_items_generated.to_string(),
            packs.pack_items_saved.to_string(),
            packs.pack_rejected_items_saved.to_string(),
            packs.pack_files_loaded.to_string(),
            packs.pack_files_written.to_string(),
            packs.pack_only_starved_jobs.to_string(),
            packs.real_pack_builds.to_string(),
            packs.tiny_packs_rejected.to_string(),
            packs.structured_topics.to_string(),
            packs.structured_subtopics.to_string(),
            packs.structured_concepts.to_string(),
            packs.structured_facts.to_string(),
            packs.structured_examples.to_string(),
            packs.structured_relations.to_string(),
            packs.extracted_training_items.to_string(),
            csv_escape(&packs.domains_touched.join("|")),
            timing.knowledge_pack_load_ms.to_string(),
            timing.knowledge_pack_save_ms.to_string(),
        ];
        csv.push_str(&fields.join(","));
        csv.push('\n');
    }

    csv
}

fn optional_u64_csv(value: Option<u64>) -> String {
    value.map(|number| number.to_string()).unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn print_benchmark_report(report: &BenchmarkReport, json_path: &str, csv_path: &str) {
    println!();
    println!("benchmark report:");
    if let Some(summary) = &report.final_summary {
        let model = &summary.final_model;
        println!("elapsed_ms={}", summary.elapsed_ms);
        println!("active_elapsed_ms={}", summary.active_elapsed_ms);
        println!("pause_ms={}", summary.pause_ms);
        println!("sleep_interruption_ms={}", summary.sleep_interruption_ms);
        println!("interruption_count={}", summary.interruption_count);
        println!("longest_sleep_interruption_ms={}", summary.longest_sleep_interruption_ms);
        println!("internal_ticks={}", summary.internal_ticks);
        println!("new_trained_items={}", summary.new_trained_items);
        println!("avg_wall_ms_per_new_trained_item={:.2}", summary.avg_wall_ms_per_new_trained_item);
        println!("avg_active_ms_per_new_trained_item={:.2}", summary.avg_active_ms_per_new_trained_item);
        println!("avg_ms_per_internal_tick={:.2}", summary.avg_ms_per_internal_tick);
        println!("reached_target={}", summary.reached_target);
        println!("stopped_by_user={}", summary.stopped_by_user);
        println!("cutoff_reason={}", summary.cutoff_reason);
        println!("acceptance_rate={:.2}%", summary.acceptance_rate * 100.0);
        println!("engraved_cases={}", model.engraved_cases);
        println!("engraved_correlations={}", model.engraved_correlations);
        println!("active_correlations={}", model.active_correlations);
        println!("preferred_paths={}", model.preferred_paths);
        println!("avg_case_confidence={:.6}", model.avg_case_confidence);
        println!("max_case_confidence={:.6}", model.max_case_confidence);
        println!("avg_correlation_score={:.6}", model.avg_correlation_score);
        println!("max_correlation_score={:.6}", model.max_correlation_score);
        println!("model_size_bytes={}", optional_u64_csv(model.model_size_bytes));
        println!("checkpoint_size_bytes={}", optional_u64_csv(model.checkpoint_size_bytes));
    }

    let dims = &report.dimension_stats;
    println!("parallel_dimensions={}", dims.parallel_dimensions);
    println!("dimension_race_items={}", dims.race_items);
    println!("dimension_unique_winner_paths={}", dims.unique_winner_paths);
    println!("dimension_unique_paths_created={}", dims.unique_paths_created);
    println!("dimension_cross_validated_paths={}", dims.cross_validated_paths);
    println!("dimension_avg_agreement_count={:.2}", dims.avg_agreement_count);
    println!("dimension_avg_winner_loss={:.6}", dims.avg_winner_loss);
    println!("dimension_best_winner_loss={:.6}", if dims.best_winner_loss == f32::MAX { 0.0 } else { dims.best_winner_loss });
    println!("dimension_worst_winner_loss={:.6}", dims.worst_winner_loss);
    println!("dimension_winner_switch_count={}", dims.winner_switch_count);

    let conv = &report.convergence_stats;
    println!("convergence_enabled={}", conv.enabled);
    println!("convergence_candidates_total={}", conv.candidates_total);
    println!("convergence_clusters_total={}", conv.clusters_total);
    println!("convergence_winner_votes_total={}", conv.winner_votes_total);
    println!("convergence_avg_winner_score={:.6}", conv.avg_winner_score);
    println!("convergence_max_winner_score={:.6}", conv.max_winner_score);
    println!("convergence_neighbor_merges={}", conv.neighbor_merges);
    println!("convergence_supporting_paths_reinforced={}", conv.supporting_paths_reinforced);
    if let Some(node) = conv.last_winner_case {
        println!("convergence_last_winner_case={}:{}:{}", node.grid, node.page, node.case);
    }

    let pre = &report.pre_final_destruction_stats;
    println!("prefinal_destruction_enabled={}", pre.enabled);
    println!("prefinal_destruction_runs={}", pre.runs);
    println!("prefinal_candidates_seen={}", pre.candidates_seen);
    println!("prefinal_candidates_forwarded={}", pre.candidates_forwarded);
    println!("prefinal_candidates_destroyed={}", pre.candidates_destroyed);
    println!("prefinal_candidates_rescued={}", pre.candidates_rescued);
    println!("prefinal_cases_destroyed={}", pre.cases_destroyed);
    println!("prefinal_correlations_destroyed={}", pre.correlations_destroyed);
    println!("prefinal_blocks_destroyed={}", pre.blocks_destroyed);
    println!("prefinal_last_alignment_score={:.6}", pre.last_alignment_score);
    if let Some(reason) = &pre.last_reason {
        println!("prefinal_last_reason={}", reason);
    }

    let timing = &report.timing;
    println!("time_subtheme_expansion_ms={}", timing.subtheme_expansion_ms);
    println!("time_training_data_generation_ms={}", timing.training_data_generation_ms);
    println!("time_validation_ms={}", timing.validation_ms);
    println!("time_model_training_ms={}", timing.model_training_ms);
    println!("time_convergence_ms={}", timing.convergence_ms);
    println!("time_checkpoint_save_ms={}", timing.checkpoint_save_ms);
    println!("time_knowledge_pack_load_ms={}", timing.knowledge_pack_load_ms);
    println!("time_knowledge_pack_save_ms={}", timing.knowledge_pack_save_ms);

    let packs = &report.knowledge_pack_stats;
    println!("knowledge_pack_enabled={}", packs.enabled);
    println!("knowledge_pack_mode={}", packs.mode);
    println!("knowledge_pack_dir={}", packs.dir);
    println!("pack_hits={}", packs.pack_hits);
    println!("pack_misses={}", packs.pack_misses);
    println!("pack_items_reused={}", packs.pack_items_reused);
    println!("pack_items_generated={}", packs.pack_items_generated);
    println!("pack_items_saved={}", packs.pack_items_saved);
    println!("pack_rejected_items_saved={}", packs.pack_rejected_items_saved);
    println!("pack_files_loaded={}", packs.pack_files_loaded);
    println!("pack_files_written={}", packs.pack_files_written);
    println!("pack_only_starved_jobs={}", packs.pack_only_starved_jobs);
    println!("real_pack_builds={}", packs.real_pack_builds);
    println!("tiny_packs_rejected={}", packs.tiny_packs_rejected);
    println!("structured_topics={}", packs.structured_topics);
    println!("structured_subtopics={}", packs.structured_subtopics);
    println!("structured_concepts={}", packs.structured_concepts);
    println!("structured_facts={}", packs.structured_facts);
    println!("structured_examples={}", packs.structured_examples);
    println!("structured_relations={}", packs.structured_relations);
    println!("extracted_training_items={}", packs.extracted_training_items);
    println!("pack_domains_touched={}", packs.domains_touched.join("|"));

    if let Some(hardware) = &report.end_hardware {
        println!("cpu_logical_cores={}", hardware.cpu_logical_cores);
        if let Some(cpu_name) = &hardware.cpu_name {
            println!("cpu_name={cpu_name}");
        }
        println!("memory_total_kb={}", optional_u64_csv(hardware.memory_total_kb));
        println!("memory_free_kb={}", optional_u64_csv(hardware.memory_free_kb));
        if let Some(gpu_name) = &hardware.gpu_name {
            println!("gpu_name={gpu_name}");
        } else {
            println!("gpu_name=");
        }
        println!("gpu_memory_total_mb={}", optional_u64_csv(hardware.gpu_memory_total_mb));
        println!("gpu_memory_used_mb={}", optional_u64_csv(hardware.gpu_memory_used_mb));
        println!("gpu_utilization_percent={}", optional_u64_csv(hardware.gpu_utilization_percent));
    }
    println!("benchmark_json={json_path}");
    println!("benchmark_csv={csv_path}");
}



fn run_knowledge_pack_stats(args: &[String]) -> Result<(), Box<dyn Error>> {
    let config = KnowledgePackConfig::from_env();
    let domain = flag_value(args, "--domain");
    let dir = Path::new(&config.dir);
    println!("Bricks AI knowledge pack stats");
    println!("dir={}", config.dir);
    println!("mode={}", config.mode);
    println!("enabled={}", config.enabled);

    if let Some(domain) = domain {
        let (pack, loaded) = load_knowledge_pack(&config, &domain)?;
        println!("domain={domain}");
        println!("loaded={loaded}");
        println!("items={}", pack.items.len());
        println!("accepted_items={}", pack.accepted_count());
        println!("rejected_items={}", pack.rejected_count());
        println!("topics={}", pack.topic_count());
        println!("subtopics={}", pack.subtopic_count());
        println!("concepts={}", pack.concept_count());
        println!("facts={}", pack.fact_count());
        println!("examples={}", pack.example_count());
        println!("relations={}", pack.relation_count());
        println!("coverage_score={:.3}", pack.coverage_score);
        println!("real_enough={}", pack.is_real_enough(&config));
        println!("path={}", knowledge_pack_path(&config, &domain).display());
        return Ok(());
    }

    if !dir.exists() {
        println!("no knowledge pack directory yet");
        return Ok(());
    }

    let mut total_items = 0usize;
    let mut total_accepted = 0usize;
    let mut total_rejected = 0usize;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let pack: KnowledgePack = serde_json::from_str(&text)?;
        total_items += pack.items.len();
        total_accepted += pack.accepted_count();
        total_rejected += pack.rejected_count();
        println!(
            "- domain={} items={} accepted={} rejected={} topics={} subtopics={} concepts={} coverage={:.3} real_enough={} path={}",
            pack.domain,
            pack.items.len(),
            pack.accepted_count(),
            pack.rejected_count(),
            pack.topic_count(),
            pack.subtopic_count(),
            pack.concept_count(),
            pack.coverage_score,
            pack.is_real_enough(&config),
            path.display()
        );
    }
    println!("total_items={total_items}");
    println!("total_accepted={total_accepted}");
    println!("total_rejected={total_rejected}");
    Ok(())
}

fn domain_from_theme_path(theme_path: &[String]) -> String {
    theme_path
        .first()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("general")
        .to_string()
}

fn sanitize_file_component(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if (ch == '-' || ch == '_' || ch.is_whitespace()) && !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() { "general".to_string() } else { trimmed }
}

fn knowledge_pack_path(config: &KnowledgePackConfig, domain: &str) -> PathBuf {
    Path::new(&config.dir).join(format!("{}.json", sanitize_file_component(domain)))
}

fn load_knowledge_pack(config: &KnowledgePackConfig, domain: &str) -> Result<(KnowledgePack, bool), Box<dyn Error>> {
    fs::create_dir_all(&config.dir)?;
    let path = knowledge_pack_path(config, domain);
    if !path.exists() {
        return Ok((KnowledgePack::new(domain), false));
    }
    let text = fs::read_to_string(path)?;
    let mut pack: KnowledgePack = serde_json::from_str(&text)?;
    if pack.domain.trim().is_empty() {
        pack.domain = domain.to_string();
    }
    Ok((pack, true))
}

fn save_knowledge_pack(config: &KnowledgePackConfig, pack: &mut KnowledgePack) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(&config.dir)?;
    pack.updated_unix_ms = unix_ms_now();
    let path = knowledge_pack_path(config, &pack.domain);
    fs::write(path, serde_json::to_string_pretty(pack)?)?;
    Ok(())
}


fn ensure_real_knowledge_pack(
    pool: &mut TeacherProviderPool,
    config: &KnowledgePackConfig,
    domain: &str,
    mut pack: KnowledgePack,
    benchmark: &mut BenchmarkRecorder,
) -> Result<(KnowledgePack, bool), Box<dyn Error>> {
    if config.continuous_only() || pack.is_real_enough(config) {
        return Ok((pack, false));
    }

    let tiny_or_legacy = pack.is_tiny_or_legacy_cache(config);
    if tiny_or_legacy {
        println!(
            "knowledge_pack domain={} rejected_tiny_cache items={} topics={} subtopics={} concepts={}; building real structured pack in chunks",
            domain,
            pack.items.len(),
            pack.topic_count(),
            pack.subtopic_count(),
            pack.concept_count()
        );
        pack = KnowledgePack::new(domain);
    } else {
        println!(
            "knowledge_pack domain={} incomplete coverage items={} topics={} subtopics={} concepts={}; enriching in chunks",
            domain,
            pack.accepted_count(),
            pack.topic_count(),
            pack.subtopic_count(),
            pack.concept_count()
        );
    }

    let mut provider_name = String::new();
    if config.local_bootstrap_first && !pack.is_real_enough(config) {
        let added = bootstrap_local_structured_knowledge_pack(&mut pack, config, domain);
        if added > 0 {
            pack.source_provider = "local_structured_bootstrap".to_string();
            provider_name = pack.source_provider.clone();
            pack.coverage_score = compute_pack_coverage_score(&pack, config);
            println!(
                "knowledge_pack domain={} local_bootstrap topics={} subtopics={} concepts={} accepted_items={} coverage={:.3}",
                domain,
                pack.topic_count(),
                pack.subtopic_count(),
                pack.concept_count(),
                pack.accepted_count(),
                pack.coverage_score
            );
        }
    }

    if config.provider_enrichment && !pack.is_real_enough(config) {
        let build_result = if config.build_chunked {
            build_real_knowledge_pack_chunked(pool, config, domain, &mut pack)
        } else {
            build_real_knowledge_pack_single_call(pool, config, domain, &mut pack)
        };

        match build_result {
            Ok(name) => provider_name = name,
            Err(error) => {
                println!(
                    "knowledge_pack domain={} provider_enrichment_failed_kept_local_structured_pack: {}; topics={} subtopics={} concepts={} accepted_items={}",
                    domain,
                    error,
                    pack.topic_count(),
                    pack.subtopic_count(),
                    pack.concept_count(),
                    pack.accepted_count()
                );
                if pack.concept_count() == 0 || pack.accepted_count() == 0 {
                    let added = bootstrap_local_structured_knowledge_pack(&mut pack, config, domain);
                    if added > 0 {
                        pack.source_provider = "local_structured_bootstrap_after_provider_failure".to_string();
                        provider_name = pack.source_provider.clone();
                    }
                }
            }
        }
    }

    if provider_name.is_empty() {
        provider_name = if pack.source_provider.trim().is_empty() {
            "local_structured_bootstrap".to_string()
        } else {
            pack.source_provider.clone()
        };
    }

    if pack.concept_count() == 0 || pack.accepted_count() == 0 {
        return Err(boxed_error(format!(
            "unable to build structured knowledge pack for {domain}; local bootstrap produced no usable data"
        )));
    }

    pack.coverage_score = compute_pack_coverage_score(&pack, config);
    if pack.curriculum_notes.trim().is_empty() {
        pack.curriculum_notes = format!(
            "Structured Bricks AI knowledge pack generated by {provider_name} using chunked local corpus building. It is used as reusable local data before live generation."
        );
    }
    let extracted_items = extract_training_items_from_structured_pack(&mut pack, config);
    pack.coverage_score = compute_pack_coverage_score(&pack, config);
    benchmark.record_real_pack_build(domain, &pack, extracted_items, tiny_or_legacy);
    println!(
        "knowledge_pack domain={} real_pack_available provider={} topics={} subtopics={} concepts={} facts={} examples={} relations={} extracted_items={} accepted_items={} coverage={:.3} real_enough={}",
        domain,
        provider_name,
        pack.topic_count(),
        pack.subtopic_count(),
        pack.concept_count(),
        pack.fact_count(),
        pack.example_count(),
        pack.relation_count(),
        extracted_items,
        pack.accepted_count(),
        pack.coverage_score,
        pack.is_real_enough(config)
    );

    if pack.accepted_count() == 0 || pack.concept_count() == 0 {
        return Err(boxed_error(format!(
            "knowledge pack for {domain} has no usable structured training data after enrichment"
        )));
    }

    Ok((pack, true))
}


fn bootstrap_local_structured_knowledge_pack(
    pack: &mut KnowledgePack,
    config: &KnowledgePackConfig,
    domain: &str,
) -> usize {
    let before = pack.concept_count();
    let mut seeded = KnowledgePack::new(domain);
    seeded.source_provider = "local_structured_bootstrap".to_string();
    seeded.curriculum_notes = format!(
        "Local structured bootstrap corpus for {domain}. It is deterministic, reusable, and designed to provide real structured training data when local Ollama cannot build a full corpus on CPU. Ollama/API enrichment can add more detail later."
    );

    let topics = fallback_topic_names(domain, config.min_topics);
    let subtopics_per_topic = ceil_div(config.min_subtopics.max(config.min_topics), config.min_topics).max(3);
    let concepts_per_subtopic = ceil_div(
        config.min_concepts.max(config.min_topics * subtopics_per_topic),
        config.min_topics * subtopics_per_topic,
    )
    .max(2);

    for topic_name in topics.into_iter().take(config.min_topics) {
        let subtopic_names = fallback_subtopic_names(domain, &topic_name, subtopics_per_topic);
        let mut topic = KnowledgeTopic {
            name: topic_name.clone(),
            summary: format!("Core structured knowledge for {topic_name} in {domain}."),
            subtopics: Vec::new(),
        };

        for subtopic_name in subtopic_names.into_iter().take(subtopics_per_topic) {
            let mut subtopic = KnowledgeSubtopic {
                name: subtopic_name.clone(),
                summary: format!("Reusable facts, definitions, examples, and relations about {subtopic_name}."),
                concepts: Vec::new(),
            };

            for index in 0..concepts_per_subtopic {
                subtopic.concepts.push(seed_concept(domain, &topic_name, &subtopic_name, index));
            }
            topic.subtopics.push(subtopic);
        }
        seeded.topics.push(topic);
    }

    normalize_knowledge_pack(&mut seeded, domain, "local_structured_bootstrap");
    merge_structured_knowledge_pack(pack, seeded);
    pack.source_provider = "local_structured_bootstrap".to_string();
    pack.coverage_score = compute_pack_coverage_score(pack, config);
    extract_training_items_from_structured_pack(pack, config);
    pack.concept_count().saturating_sub(before)
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    if divisor == 0 {
        value
    } else {
        value.div_ceil(divisor)
    }
}

fn fallback_subtopic_names(domain: &str, topic: &str, count: usize) -> Vec<String> {
    let lower_domain = domain.to_ascii_lowercase();
    let lower_topic = topic.to_ascii_lowercase();
    let mut names: Vec<String> = if lower_domain.contains("science") {
        science_subtopics(&lower_topic)
    } else if lower_domain.contains("math") {
        math_subtopics(&lower_topic)
    } else if lower_domain.contains("computer") {
        computer_science_subtopics(&lower_topic)
    } else if lower_domain.contains("agriculture") {
        agriculture_subtopics(&lower_topic)
    } else {
        vec![
            format!("{topic} foundations"),
            format!("{topic} methods"),
            format!("{topic} applications"),
            format!("{topic} evaluation"),
        ]
    };
    while names.len() < count {
        names.push(format!("{topic} structured area {}", names.len() + 1));
    }
    names
}

fn science_subtopics(topic: &str) -> Vec<String> {
    if topic.contains("atomic") || topic.contains("matter") {
        vec!["Protons, neutrons, and electrons", "Atomic number and isotopes", "Electron shells and bonding"]
    } else if topic.contains("cell") || topic.contains("life") {
        vec!["Cell membrane and organelles", "DNA and protein synthesis", "Cell division and specialization"]
    } else if topic.contains("evolution") {
        vec!["Natural selection", "Mutation and variation", "Speciation and adaptation"]
    } else if topic.contains("genetic") {
        vec!["Genes and alleles", "Inheritance patterns", "Genotype and phenotype"]
    } else if topic.contains("neuro") {
        vec!["Neurons and synapses", "Brain regions", "Signals and behavior"]
    } else if topic.contains("photo") {
        vec!["Light reactions", "Carbon fixation", "Chloroplast structure"]
    } else if topic.contains("quantum") {
        vec!["Wave-particle duality", "Energy levels", "Measurement and probability"]
    } else if topic.contains("system") {
        vec!["Feedback loops", "Emergent behavior", "Modeling interactions"]
    } else {
        vec!["Observation and measurement", "Evidence and models", "Causality and uncertainty"]
    }
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn math_subtopics(topic: &str) -> Vec<String> {
    if topic.contains("algebra") {
        vec!["Variables and expressions", "Equations and inequalities", "Functions and graphs"]
    } else if topic.contains("geometry") {
        vec!["Shapes and angles", "Coordinate geometry", "Area and volume"]
    } else if topic.contains("calculus") {
        vec!["Limits", "Derivatives", "Integrals"]
    } else if topic.contains("probability") {
        vec!["Events and sample spaces", "Conditional probability", "Expected value"]
    } else if topic.contains("statistics") {
        vec!["Descriptive statistics", "Sampling", "Correlation and regression"]
    } else {
        vec!["Definitions and notation", "Problem solving", "Proof and verification"]
    }
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn computer_science_subtopics(topic: &str) -> Vec<String> {
    if topic.contains("algorithm") {
        vec!["Complexity", "Search and sorting", "Correctness"]
    } else if topic.contains("data") {
        vec!["Arrays and lists", "Trees and graphs", "Hash maps"]
    } else if topic.contains("network") {
        vec!["Protocols", "Routing", "Reliability"]
    } else if topic.contains("security") {
        vec!["Authentication", "Input validation", "Threat modeling"]
    } else {
        vec!["Representation", "Execution", "Testing"]
    }
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn agriculture_subtopics(topic: &str) -> Vec<String> {
    if topic.contains("soil") {
        vec!["Soil texture", "Soil fertility", "Soil water"]
    } else if topic.contains("crop") {
        vec!["Crop selection", "Planting density", "Yield factors"]
    } else if topic.contains("irrigation") {
        vec!["Water demand", "Drip irrigation", "Drainage"]
    } else if topic.contains("pest") {
        vec!["Integrated pest management", "Monitoring", "Biological controls"]
    } else {
        vec!["Production planning", "Resource management", "Quality control"]
    }
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn seed_concept(domain: &str, topic: &str, subtopic: &str, index: usize) -> KnowledgeConcept {
    let (name, definition, facts, examples, relation_target) = domain_seed_content(domain, topic, subtopic, index);
    KnowledgeConcept {
        name: name.clone(),
        definition,
        difficulty: if index == 0 { "basic".to_string() } else { "intermediate".to_string() },
        facts,
        examples,
        qa_pairs: vec![
            KnowledgeQaPair {
                question: format!("What is the role of {name} in {domain}?"),
                answer: format!("{name} helps explain {subtopic} within {topic} and connects observations to reusable knowledge."),
            },
            KnowledgeQaPair {
                question: format!("Why does {name} matter?"),
                answer: format!("It matters because it links the initial context in {domain} to reliable conclusions, examples, and validation paths."),
            },
        ],
        relations: vec![KnowledgeRelation {
            relation_type: "part_of".to_string(),
            target: relation_target,
            description: format!("{name} is part of {subtopic} and contributes to the broader topic {topic}."),
        }],
        source_notes: "Local bootstrap based on widely established textbook-level public knowledge; intended for offline seeding before provider enrichment.".to_string(),
        tags: vec![domain.to_string(), topic.to_string(), subtopic.to_string()],
    }
}

fn domain_seed_content(
    domain: &str,
    topic: &str,
    subtopic: &str,
    index: usize,
) -> (String, String, Vec<String>, Vec<String>, String) {
    let lower_domain = domain.to_ascii_lowercase();
    let lower_topic = topic.to_ascii_lowercase();
    let lower_subtopic = subtopic.to_ascii_lowercase();
    if lower_domain.contains("science") {
        science_seed_content(topic, subtopic, &lower_topic, &lower_subtopic, index)
    } else if lower_domain.contains("math") {
        math_seed_content(topic, subtopic, &lower_topic, &lower_subtopic, index)
    } else if lower_domain.contains("computer") {
        computer_seed_content(topic, subtopic, &lower_topic, &lower_subtopic, index)
    } else if lower_domain.contains("agriculture") {
        agriculture_seed_content(topic, subtopic, &lower_topic, &lower_subtopic, index)
    } else {
        let name = if index == 0 { subtopic.to_string() } else { format!("{} validation", subtopic) };
        (
            name.clone(),
            format!("{name} is a structured concept in {domain} used to connect definitions, facts, examples, and evaluation."),
            vec![
                format!("Reliable work in {domain} separates observations from interpretations."),
                format!("{topic} knowledge becomes useful when it can be checked against examples and constraints."),
                format!("{subtopic} can be evaluated by consistency, evidence, and practical consequences."),
            ],
            vec![format!("A concrete {domain} case can be analyzed by identifying the context, the method, and the result.")],
            topic.to_string(),
        )
    }
}

fn science_seed_content(
    topic: &str,
    subtopic: &str,
    lower_topic: &str,
    lower_subtopic: &str,
    index: usize,
) -> (String, String, Vec<String>, Vec<String>, String) {
    if lower_subtopic.contains("proton") || lower_topic.contains("atomic") {
        let name = if index == 0 { "Atomic structure" } else { "Electron configuration" };
        return (
            name.to_string(),
            "Atomic structure describes how protons, neutrons, and electrons form atoms and determine chemical behavior.".to_string(),
            vec![
                "The atomic number of an element equals the number of protons in its nucleus.".to_string(),
                "Isotopes of the same element have the same number of protons but different numbers of neutrons.".to_string(),
                "Electron arrangements influence bonding, reactivity, and periodic trends.".to_string(),
            ],
            vec!["Carbon has atomic number 6, so every neutral carbon atom has 6 protons and 6 electrons.".to_string()],
            "Matter".to_string(),
        );
    }
    if lower_subtopic.contains("cell") || lower_topic.contains("cell") {
        let name = if index == 0 { "Cell membrane" } else { "Organelle function" };
        return (
            name.to_string(),
            "Cell biology studies cells as the basic units of life, including membranes, organelles, genetic material, and metabolism.".to_string(),
            vec![
                "The cell membrane regulates movement of substances into and out of the cell.".to_string(),
                "DNA stores genetic instructions that cells use to make RNA and proteins.".to_string(),
                "Mitochondria release usable energy from nutrients in many eukaryotic cells.".to_string(),
            ],
            vec!["A plant leaf cell contains chloroplasts that help convert light energy into chemical energy.".to_string()],
            "Life Systems".to_string(),
        );
    }
    if lower_topic.contains("evolution") || lower_subtopic.contains("selection") {
        let name = if index == 0 { "Natural selection" } else { "Genetic variation" };
        return (
            name.to_string(),
            "Evolution describes changes in heritable traits of populations over generations.".to_string(),
            vec![
                "Natural selection can increase traits that improve survival or reproduction in a given environment.".to_string(),
                "Mutation and recombination create genetic variation in populations.".to_string(),
                "Evolution acts on populations across generations, not on a single individual instantly.".to_string(),
            ],
            vec!["Bacteria with resistance traits can become more common after exposure to an antibiotic.".to_string()],
            "Genetics".to_string(),
        );
    }
    if lower_topic.contains("photo") {
        let name = if index == 0 { "Photosynthesis" } else { "Carbon fixation" };
        return (
            name.to_string(),
            "Photosynthesis converts light energy into chemical energy stored in sugars.".to_string(),
            vec![
                "Plants use carbon dioxide, water, and light energy to produce sugars and release oxygen.".to_string(),
                "Chlorophyll absorbs light used in the light-dependent reactions.".to_string(),
                "Carbon fixation incorporates carbon dioxide into organic molecules.".to_string(),
            ],
            vec!["A green leaf exposed to sunlight can produce sugars used for growth and storage.".to_string()],
            "Energy".to_string(),
        );
    }
    let name = if index == 0 { subtopic.to_string() } else { format!("{} model", subtopic) };
    (
        name.clone(),
        format!("{name} is a scientific concept in {topic} used to connect observation, measurement, models, and evidence."),
        vec![
            "Scientific claims are strengthened by reproducible observations and clear measurement methods.".to_string(),
            "Models simplify complex systems so predictions can be tested against evidence.".to_string(),
            "Uncertainty is reduced by repeated measurements, controls, and comparison with alternatives.".to_string(),
        ],
        vec![format!("A scientist can test a {subtopic} claim by defining variables, collecting measurements, and comparing results.")],
        topic.to_string(),
    )
}

fn math_seed_content(
    topic: &str,
    subtopic: &str,
    lower_topic: &str,
    _lower_subtopic: &str,
    index: usize,
) -> (String, String, Vec<String>, Vec<String>, String) {
    let name = if index == 0 { subtopic.to_string() } else { format!("{} procedure", subtopic) };
    let (definition, facts, examples) = if lower_topic.contains("algebra") {
        (
            "Algebra represents quantities and relationships with symbols so unknown values can be solved or generalized.".to_string(),
            vec![
                "An equation states that two expressions have equal value.".to_string(),
                "Operations applied to both sides of an equation preserve equality when they are valid.".to_string(),
                "A function maps each input in its domain to exactly one output.".to_string(),
            ],
            vec!["Solving 2x + 3 = 11 gives x = 4 after subtracting 3 and dividing by 2.".to_string()],
        )
    } else if lower_topic.contains("calculus") {
        (
            "Calculus studies change and accumulation using limits, derivatives, and integrals.".to_string(),
            vec![
                "A derivative measures instantaneous rate of change.".to_string(),
                "An integral can represent accumulated quantity or area under a curve.".to_string(),
                "Limits describe behavior as inputs approach a value.".to_string(),
            ],
            vec!["Velocity is the derivative of position with respect to time.".to_string()],
        )
    } else {
        (
            format!("{name} organizes mathematical objects, rules, and transformations for precise reasoning."),
            vec![
                "Mathematical definitions set exact conditions for using a concept.".to_string(),
                "Examples and counterexamples help test whether a statement is generally true.".to_string(),
                "Proof connects assumptions to conclusions through valid logical steps.".to_string(),
            ],
            vec![format!("A {topic} problem can be solved by identifying givens, choosing a rule, and checking the result.")],
        )
    };
    (name, definition, facts, examples, topic.to_string())
}

fn computer_seed_content(
    topic: &str,
    subtopic: &str,
    lower_topic: &str,
    _lower_subtopic: &str,
    index: usize,
) -> (String, String, Vec<String>, Vec<String>, String) {
    let name = if index == 0 { subtopic.to_string() } else { format!("{} tradeoff", subtopic) };
    let (definition, facts, examples) = if lower_topic.contains("algorithm") {
        (
            "Algorithms are finite procedures for transforming inputs into outputs.".to_string(),
            vec![
                "Time complexity describes how running time grows with input size.".to_string(),
                "Correctness means an algorithm returns the intended result for valid inputs.".to_string(),
                "Different algorithms can solve the same problem with different tradeoffs.".to_string(),
            ],
            vec!["Binary search finds a target in a sorted list by repeatedly halving the search interval.".to_string()],
        )
    } else {
        (
            format!("{name} is a computer science concept used to represent, process, secure, or communicate information."),
            vec![
                "Programs transform data according to explicit instructions.".to_string(),
                "Testing compares expected behavior with actual behavior.".to_string(),
                "Security improves when inputs are validated and privileges are limited.".to_string(),
            ],
            vec!["A hash map can store key-value pairs for fast lookup by key.".to_string()],
        )
    };
    (name, definition, facts, examples, topic.to_string())
}

fn agriculture_seed_content(
    topic: &str,
    subtopic: &str,
    lower_topic: &str,
    _lower_subtopic: &str,
    index: usize,
) -> (String, String, Vec<String>, Vec<String>, String) {
    let name = if index == 0 { subtopic.to_string() } else { format!("{} management", subtopic) };
    let (definition, facts, examples) = if lower_topic.contains("soil") {
        (
            "Soil science studies soil physical structure, nutrients, organisms, water, and suitability for crops.".to_string(),
            vec![
                "Soil texture depends on proportions of sand, silt, and clay.".to_string(),
                "Organic matter can improve soil structure, water retention, and nutrient availability.".to_string(),
                "Soil pH influences nutrient availability and crop suitability.".to_string(),
            ],
            vec!["A loam soil often supports many crops because it balances drainage and water retention.".to_string()],
        )
    } else {
        (
            format!("{name} supports agricultural production by connecting biological needs, resources, timing, and yield outcomes."),
            vec![
                "Crop yield is affected by genetics, soil, water, nutrients, pests, weather, and management.".to_string(),
                "Irrigation planning balances crop water demand with soil drainage and water availability.".to_string(),
                "Integrated pest management combines monitoring, prevention, biological controls, and targeted treatment.".to_string(),
            ],
            vec!["A farmer may adjust planting density to balance light, nutrients, disease risk, and expected yield.".to_string()],
        )
    };
    (name, definition, facts, examples, topic.to_string())
}

fn build_real_knowledge_pack_single_call(
    pool: &mut TeacherProviderPool,
    config: &KnowledgePackConfig,
    domain: &str,
    pack: &mut KnowledgePack,
) -> Result<String, String> {
    let prompt = build_real_knowledge_pack_prompt(config, domain);
    let (provider_name, mut generated_pack) = pool.generate_parsed(&prompt, |text| {
        parse_knowledge_pack_object(text, domain).map_err(|error| error.to_string())
    })?;
    normalize_knowledge_pack(&mut generated_pack, domain, &provider_name);
    merge_structured_knowledge_pack(pack, generated_pack);
    pack.source_provider = provider_name.clone();
    Ok(provider_name)
}

fn build_real_knowledge_pack_chunked(
    pool: &mut TeacherProviderPool,
    config: &KnowledgePackConfig,
    domain: &str,
    pack: &mut KnowledgePack,
) -> Result<String, String> {
    let mut provider_name = String::new();
    let mut topic_names = existing_topic_names(pack);
    if topic_names.len() < config.min_topics {
        match generate_topic_names(pool, domain, config.min_topics) {
            Ok((name, generated_topics)) => {
                provider_name = name;
                for topic in generated_topics {
                    if !topic_names.iter().any(|existing| existing.eq_ignore_ascii_case(&topic)) {
                        topic_names.push(topic);
                    }
                }
            }
            Err(error) => {
                println!("knowledge_pack domain={domain} topic_list_fallback reason={error}");
                for topic in fallback_topic_names(domain, config.min_topics) {
                    if !topic_names.iter().any(|existing| existing.eq_ignore_ascii_case(&topic)) {
                        topic_names.push(topic);
                    }
                }
            }
        }
    }

    let mut built_chunks = 0usize;
    let mut chunk_errors = Vec::new();
    for topic_name in topic_names {
        if built_chunks >= config.max_build_chunks || pack.is_real_enough(config) {
            break;
        }
        if topic_is_complete(pack, &topic_name, config) {
            continue;
        }
        let prompt = build_topic_knowledge_pack_prompt(config, domain, &topic_name);
        match pool.generate_parsed(&prompt, |text| {
            parse_knowledge_pack_object(text, domain).map_err(|error| error.to_string())
        }) {
            Ok((name, mut topic_pack)) => {
                provider_name = name.clone();
                normalize_knowledge_pack(&mut topic_pack, domain, &name);
                merge_structured_knowledge_pack(pack, topic_pack);
                pack.source_provider = name;
                let before = pack.accepted_count();
                let extracted = extract_training_items_from_structured_pack(pack, config);
                pack.coverage_score = compute_pack_coverage_score(pack, config);
                built_chunks += 1;
                println!(
                    "knowledge_pack domain={} topic_chunk={} built={} topics={} subtopics={} concepts={} new_items={} accepted_items={} coverage={:.3}",
                    domain,
                    topic_name,
                    built_chunks,
                    pack.topic_count(),
                    pack.subtopic_count(),
                    pack.concept_count(),
                    pack.accepted_count().saturating_sub(before).max(extracted),
                    pack.accepted_count(),
                    pack.coverage_score
                );
            }
            Err(error) => {
                let msg = format!("topic `{topic_name}` failed: {error}");
                println!("knowledge_pack domain={domain} topic_chunk_failed {msg}");
                chunk_errors.push(msg);
            }
        }
    }

    if pack.concept_count() > 0 && pack.accepted_count() > 0 {
        if provider_name.is_empty() {
            provider_name = "partial_structured_pack".to_string();
        }
        Ok(provider_name)
    } else {
        Err(chunk_errors.join("\n"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct KnowledgeTopicList {
    #[serde(default)]
    topics: Vec<String>,
}

fn generate_topic_names(
    pool: &mut TeacherProviderPool,
    domain: &str,
    min_topics: usize,
) -> Result<(String, Vec<String>), String> {
    let prompt = format!(
        "Return ONLY JSON, no markdown. Build a curriculum topic list for the domain `{domain}`. Return at least {min_topics} broad, concrete, reusable knowledge topics. Shape: {{\"topics\":[\"Topic 1\",\"Topic 2\"]}}. Do not include questions."
    );
    pool.generate_parsed(&prompt, |text| {
        let parsed: KnowledgeTopicList = parse_json_object(text).map_err(|error| error.to_string())?;
        let topics = parsed
            .topics
            .into_iter()
            .map(|topic| topic.trim().to_string())
            .filter(|topic| !topic.is_empty())
            .collect::<Vec<_>>();
        if topics.is_empty() {
            Err("topic list was empty".to_string())
        } else {
            Ok(topics)
        }
    })
}

fn existing_topic_names(pack: &KnowledgePack) -> Vec<String> {
    pack.topics.iter().map(|topic| topic.name.clone()).collect()
}

fn topic_is_complete(pack: &KnowledgePack, topic_name: &str, config: &KnowledgePackConfig) -> bool {
    let min_subtopics = config.chunk_subtopics_per_topic;
    let min_concepts = config.chunk_subtopics_per_topic * config.chunk_concepts_per_subtopic;
    pack.topics.iter().any(|topic| {
        topic.name.eq_ignore_ascii_case(topic_name)
            && topic.subtopics.len() >= min_subtopics
            && topic.subtopics.iter().map(|subtopic| subtopic.concepts.len()).sum::<usize>() >= min_concepts
    })
}

fn fallback_topic_names(domain: &str, min_topics: usize) -> Vec<String> {
    let lower = domain.to_ascii_lowercase();
    let base: Vec<&str> = if lower.contains("science") {
        vec!["Scientific Method", "Measurement", "Matter", "Energy", "Forces", "Earth Systems", "Life Systems", "Probability and Statistics"]
    } else if lower.contains("math") {
        vec!["Arithmetic", "Algebra", "Geometry", "Trigonometry", "Calculus", "Probability", "Statistics", "Discrete Mathematics"]
    } else if lower.contains("agriculture") {
        vec!["Soil Science", "Crop Production", "Irrigation", "Plant Nutrition", "Pest Management", "Harvesting", "Livestock Systems", "Agricultural Economics"]
    } else if lower.contains("computer") {
        vec!["Algorithms", "Data Structures", "Operating Systems", "Databases", "Networks", "Programming Languages", "Software Engineering", "Security Fundamentals"]
    } else {
        vec!["Foundations", "Core Concepts", "Methods", "Systems", "Tools", "Applications", "Evaluation", "History and Context"]
    };
    let mut topics = base.into_iter().map(str::to_string).collect::<Vec<_>>();
    while topics.len() < min_topics {
        topics.push(format!("{} Topic {}", domain, topics.len() + 1));
    }
    topics
}

fn build_topic_knowledge_pack_prompt(config: &KnowledgePackConfig, domain: &str, topic: &str) -> String {
    let subtopics = config.chunk_subtopics_per_topic;
    let concepts = config.chunk_concepts_per_subtopic;
    format!(
        "Return ONLY one valid JSON object, no markdown. Build ONE compact but real structured Bricks AI knowledge-pack chunk. Domain: {domain}. Topic: {topic}. Include exactly one topic named `{topic}` with at least {subtopics} subtopics and at least {concepts} concepts per subtopic. Each concept must include: definition, 2 facts, 1 concrete example, 1 question-answer pair, 1 relation, difficulty, tags, source_notes. Keep answers factual and concise. Do not return a tiny quiz cache. JSON shape: {{\"format\":\"bricks-ai-knowledge-pack\",\"version\":2,\"domain\":\"{domain}\",\"coverage_score\":0.0,\"curriculum_notes\":\"chunked structured corpus\",\"topics\":[{{\"name\":\"{topic}\",\"summary\":\"...\",\"subtopics\":[{{\"name\":\"...\",\"summary\":\"...\",\"concepts\":[{{\"name\":\"...\",\"definition\":\"...\",\"difficulty\":\"basic\",\"facts\":[\"...\",\"...\"],\"examples\":[\"...\"],\"qa_pairs\":[{{\"question\":\"...\",\"answer\":\"...\"}}],\"relations\":[{{\"relation_type\":\"depends_on\",\"target\":\"...\",\"description\":\"...\"}}],\"source_notes\":\"general textbook or public reference knowledge\",\"tags\":[\"{domain}\",\"{topic}\"]}}]}}]}}]}}"
    )
}


fn build_real_knowledge_pack_prompt(config: &KnowledgePackConfig, domain: &str) -> String {
    let topics = config.min_topics;
    let build_depth = config.build_depth;
    let subtopics_per_topic = (config.min_subtopics / topics.max(1)).max(2);
    let concepts_per_subtopic = (config.min_concepts / config.min_subtopics.max(1)).max(2);
    format!(
        "You are building a real reusable Bricks AI knowledge pack, not a tiny quiz cache.\n\nDomain: {domain}\n\nReturn ONLY one valid JSON object, no markdown. Build a structured curriculum with depth {build_depth} and at least {topics} broad topics. Each topic must have at least {subtopics_per_topic} subtopics. Each subtopic must have at least {concepts_per_subtopic} concepts. Each concept must include a clear definition, 3 factual statements, 2 concrete examples, 2 question-answer pairs, at least 1 relation to another concept, difficulty, tags, and source_notes.\n\nThe JSON shape must be exactly compatible with this schema:\n{{\"format\":\"bricks-ai-knowledge-pack\",\"version\":2,\"domain\":\"{domain}\",\"coverage_score\":0.0,\"curriculum_notes\":\"...\",\"topics\":[{{\"name\":\"...\",\"summary\":\"...\",\"subtopics\":[{{\"name\":\"...\",\"summary\":\"...\",\"concepts\":[{{\"name\":\"...\",\"definition\":\"...\",\"difficulty\":\"basic|intermediate|advanced\",\"facts\":[\"...\"],\"examples\":[\"...\"],\"qa_pairs\":[{{\"question\":\"...\",\"answer\":\"...\"}}],\"relations\":[{{\"relation_type\":\"depends_on|part_of|contrasts_with|applies_to\",\"target\":\"...\",\"description\":\"...\"}}],\"source_notes\":\"Use general established knowledge; mention source family such as textbook, public documentation, or scientific consensus where relevant.\",\"tags\":[\"{domain}\"]}}]}}]}}]}}\n\nDo not return one isolated question. Build a reusable local corpus. Avoid personal advice and high-stakes instructions."
    )
}

fn parse_knowledge_pack_object(text: &str, domain: &str) -> Result<KnowledgePack, Box<dyn Error>> {
    let mut pack: KnowledgePack = parse_json_object(text)?;
    if pack.domain.trim().is_empty() {
        pack.domain = domain.to_string();
    }
    Ok(pack)
}

fn normalize_knowledge_pack(pack: &mut KnowledgePack, domain: &str, provider_name: &str) {
    let now = unix_ms_now();
    if pack.format.trim().is_empty() {
        pack.format = "bricks-ai-knowledge-pack".to_string();
    }
    if pack.version < 2 {
        pack.version = 2;
    }
    if pack.domain.trim().is_empty() {
        pack.domain = domain.to_string();
    }
    if pack.created_unix_ms == 0 {
        pack.created_unix_ms = now;
    }
    pack.updated_unix_ms = now;
    if pack.source_provider.trim().is_empty() {
        pack.source_provider = provider_name.to_string();
    }
    for topic in &mut pack.topics {
        topic.name = topic.name.trim().to_string();
        for subtopic in &mut topic.subtopics {
            subtopic.name = subtopic.name.trim().to_string();
            for concept in &mut subtopic.concepts {
                concept.name = concept.name.trim().to_string();
                concept.facts.retain(|value| !value.trim().is_empty());
                concept.examples.retain(|value| !value.trim().is_empty());
                concept.qa_pairs.retain(|qa| !qa.question.trim().is_empty() && !qa.answer.trim().is_empty());
                concept.relations.retain(|rel| !rel.target.trim().is_empty() || !rel.description.trim().is_empty());
                if concept.difficulty.trim().is_empty() {
                    concept.difficulty = "basic".to_string();
                }
                if concept.tags.is_empty() {
                    concept.tags.push(domain.to_ascii_lowercase());
                }
            }
            subtopic.concepts.retain(|concept| !concept.name.trim().is_empty());
        }
        topic.subtopics.retain(|subtopic| !subtopic.name.trim().is_empty());
    }
    pack.topics.retain(|topic| !topic.name.trim().is_empty());
}

fn merge_structured_knowledge_pack(target: &mut KnowledgePack, source: KnowledgePack) {
    if target.domain.trim().is_empty() {
        target.domain = source.domain.clone();
    }
    if target.curriculum_notes.trim().is_empty() {
        target.curriculum_notes = source.curriculum_notes.clone();
    }
    target.source_provider = source.source_provider.clone();
    let mut existing_topics: HashSet<String> = target.topics.iter().map(|topic| topic.name.to_ascii_lowercase()).collect();
    for topic in source.topics {
        let key = topic.name.to_ascii_lowercase();
        if existing_topics.insert(key) {
            target.topics.push(topic);
        }
    }
    let mut existing_items: HashSet<String> = target.items.iter().map(|item| item.id.clone()).collect();
    for item in source.items {
        if existing_items.insert(item.id.clone()) {
            target.items.push(item);
        }
    }
}

fn extract_training_items_from_structured_pack(pack: &mut KnowledgePack, config: &KnowledgePackConfig) -> usize {
    let now = unix_ms_now();
    let mut existing: HashSet<String> = pack.items.iter().map(|item| item.id.clone()).collect();
    let mut generated: Vec<KnowledgePackItem> = Vec::new();

    for topic in &pack.topics {
        for subtopic in &topic.subtopics {
            for concept in &subtopic.concepts {
                if generated.len() + pack.accepted_count() >= config.min_items_per_domain {
                    break;
                }
                let theme_path = vec![pack.domain.clone(), topic.name.clone(), subtopic.name.clone()];
                let mut push_item = |question: String, answer: String, tags: Vec<String>, notes: String| {
                    if question.trim().is_empty() || answer.trim().is_empty() {
                        return;
                    }
                    let id = format!("{:016x}", stable_hash64(&format!(
                        "{}|{}|{}",
                        theme_path.join("/"),
                        question.trim(),
                        answer.trim()
                    )));
                    if !existing.insert(id.clone()) {
                        return;
                    }
                    generated.push(KnowledgePackItem {
                        id,
                        theme_path: theme_path.clone(),
                        question,
                        answer,
                        teacher_confidence: 0.92,
                        validator_score: 0.95,
                        accepted: true,
                        tags,
                        verification_notes: notes,
                        source_provider: pack.source_provider.clone(),
                        validator_provider: "structured_pack_extractor".to_string(),
                        created_unix_ms: now,
                        used_count: 0,
                        last_used_unix_ms: None,
                    });
                };

                if !concept.definition.trim().is_empty() {
                    push_item(
                        format!("What is {} in {}?", concept.name, pack.domain),
                        concept.definition.clone(),
                        concept.tags.clone(),
                        concept.source_notes.clone(),
                    );
                }
                for fact in concept.facts.iter().take(config.extraction_items_per_concept) {
                    push_item(
                        format!("What is an important fact about {}?", concept.name),
                        fact.clone(),
                        concept.tags.clone(),
                        concept.source_notes.clone(),
                    );
                }
                for example in concept.examples.iter().take(config.extraction_items_per_concept) {
                    push_item(
                        format!("Give a concrete example of {}.", concept.name),
                        example.clone(),
                        concept.tags.clone(),
                        concept.source_notes.clone(),
                    );
                }
                for qa in concept.qa_pairs.iter().take(config.extraction_items_per_concept) {
                    push_item(
                        qa.question.clone(),
                        qa.answer.clone(),
                        concept.tags.clone(),
                        concept.source_notes.clone(),
                    );
                }
                for relation in concept.relations.iter().take(config.extraction_items_per_concept) {
                    let answer = if relation.description.trim().is_empty() {
                        format!("{} is related to {} by {}.", concept.name, relation.target, relation.relation_type)
                    } else {
                        relation.description.clone()
                    };
                    push_item(
                        format!("How is {} related to {}?", concept.name, relation.target),
                        answer,
                        concept.tags.clone(),
                        concept.source_notes.clone(),
                    );
                }
            }
        }
    }

    let created = generated.len();
    pack.items.extend(generated);
    created
}

fn compute_pack_coverage_score(pack: &KnowledgePack, config: &KnowledgePackConfig) -> f32 {
    let topics = ratio_capped(pack.topic_count(), config.min_topics);
    let subtopics = ratio_capped(pack.subtopic_count(), config.min_subtopics);
    let concepts = ratio_capped(pack.concept_count(), config.min_concepts);
    let items = ratio_capped(pack.accepted_count(), config.min_items_per_domain);
    ((topics + subtopics + concepts + items) / 4.0).clamp(0.0, 1.0)
}

fn ratio_capped(value: usize, target: usize) -> f32 {
    if target == 0 {
        1.0
    } else {
        (value as f32 / target as f32).clamp(0.0, 1.0)
    }
}

fn take_knowledge_items_for_job(
    pack: &mut KnowledgePack,
    job: &PackJob,
    limit: usize,
    max_reuse_per_item: usize,
) -> Vec<ValidatedPackItem> {
    let mut selected = Vec::new();
    let now = unix_ms_now();
    for item in &mut pack.items {
        if selected.len() >= limit {
            break;
        }
        if !item.accepted || item.used_count >= max_reuse_per_item {
            continue;
        }
        if !theme_matches(&item.theme_path, &job.theme_path) {
            continue;
        }
        item.used_count += 1;
        item.last_used_unix_ms = Some(now);
        selected.push(ValidatedPackItem {
            theme_path: item.theme_path.clone(),
            question: item.question.clone(),
            answer: item.answer.clone(),
            teacher_confidence: item.teacher_confidence,
            validator_score: item.validator_score,
            accepted: true,
        });
    }
    selected
}

fn theme_matches(item_path: &[String], job_path: &[String]) -> bool {
    item_path == job_path
        || item_path.starts_with(job_path)
        || job_path.starts_with(item_path)
        || item_path.first() == job_path.first()
}

fn append_generated_items_to_knowledge_pack(
    pack: &mut KnowledgePack,
    job: &PackJob,
    items: &[ApiTeacherItem],
    validation_scores: &HashMap<usize, f32>,
    generator_name: &str,
    validator_name: &str,
    trainer: &RawAITrainer,
) -> (usize, usize) {
    let mut accepted_saved = 0usize;
    let mut rejected_saved = 0usize;
    let now = unix_ms_now();
    let mut existing: HashSet<String> = pack.items.iter().map(|item| item.id.clone()).collect();

    for (index, item) in items.iter().enumerate() {
        let validator_score = validation_scores.get(&index).copied().unwrap_or(0.0);
        let accepted = trainer.validate_pack_item(job, item.clone().into_core(), validator_score).accepted;
        let id = format!("{:016x}", stable_hash64(&format!(
            "{}|{}|{}",
            job.theme_path.join("/"),
            item.question.trim(),
            item.answer.trim()
        )));
        if existing.contains(&id) {
            continue;
        }
        existing.insert(id.clone());
        if accepted {
            accepted_saved += 1;
        } else {
            rejected_saved += 1;
        }
        pack.items.push(KnowledgePackItem {
            id,
            theme_path: job.theme_path.clone(),
            question: item.question.clone(),
            answer: item.answer.clone(),
            teacher_confidence: item.confidence.clamp(0.0, 1.0),
            validator_score,
            accepted,
            tags: item.tags.clone(),
            verification_notes: item.verification_notes.clone(),
            source_provider: generator_name.to_string(),
            validator_provider: validator_name.to_string(),
            created_unix_ms: now,
            used_count: 0,
            last_used_unix_ms: None,
        });
    }

    (accepted_saved, rejected_saved)
}

fn train_validated_item_through_dimensions(
    trainer: &mut RawAITrainer,
    benchmark: &mut BenchmarkRecorder,
    internal_tick: usize,
    theme_path: &[String],
    dimension_config: &DimensionConfig,
    convergence_config: &ConvergenceConfig,
    pre_final_config: &PreFinalDestructionConfig,
    validated: &ValidatedPackItem,
    source_label: &str,
) -> bool {
    if !validated.accepted {
        trainer.pack_state.rejected_items += 1;
        return false;
    }

    let convergence_started = Instant::now();
    if let Some(result) = train_with_parallel_dimensions(trainer, validated, dimension_config, convergence_config, pre_final_config) {
        benchmark.add_timing("convergence", convergence_started.elapsed());
        benchmark.record_pre_final_destruction(&result.pre_final_stats);
        benchmark.record_dimension_result(internal_tick, theme_path, &result);
        println!(
            "trained item: source={}, loss={:.6}, dimension={}, score={:.4}, conv_score={:.4}, conv_votes={}, final={}:{}:{}, agreement={}/{}, cross_validated={}, pre_destroyed={}, teacher_confidence={:.2}, validator_score={:.2}",
            source_label,
            result.winner_loss,
            result.winner_dimension_id,
            result.winner_score,
            result.convergence_score,
            result.convergence_votes,
            result.convergence_final_node.grid,
            result.convergence_final_node.page,
            result.convergence_final_node.case,
            result.agreement_count,
            result.candidate_count,
            result.cross_validated_paths,
            result.pre_final_stats.candidates_destroyed,
            validated.teacher_confidence,
            validated.validator_score
        );
        true
    } else {
        false
    }
}

fn run_teacher_pack_training_local(args: &[String]) -> Result<(), Box<dyn Error>> {
    let steps = flag_usize(args, "--steps")
        .or_else(|| env_usize("BRICKS_AI_STEPS"))
        .unwrap_or(8);

    let items_per_theme = flag_usize(args, "--items")
        .or_else(|| env_usize("BRICKS_AI_ITEMS_PER_THEME"))
        .unwrap_or(3);

    let depth = flag_usize(args, "--depth")
        .or_else(|| env_usize("BRICKS_AI_MAX_DEPTH"))
        .unwrap_or(1);

    let checkpoint_path = flag_value(args, "--checkpoint")
        .or_else(|| env::var("BRICKS_AI_CHECKPOINT").ok())
        .unwrap_or_else(|| "bricks_ai_checkpoint.bin".to_string());

    let resume = flag_present(args, "--resume") || env_bool("BRICKS_AI_RESUME").unwrap_or(false);
    let extend = flag_present(args, "--extend") || env_bool("BRICKS_AI_EXTEND").unwrap_or(false);
    let auto_extend_empty_resume = env_bool("BRICKS_AI_AUTO_EXTEND_ON_EMPTY_RESUME").unwrap_or(true);

    let device = detect_local_compute();
    let dimension_config = DimensionConfig::from_env();
    let convergence_config = ConvergenceConfig::from_env();
    let pre_final_config = PreFinalDestructionConfig::from_env();
    let knowledge_config = KnowledgePackConfig::from_env();
    let mut pool = TeacherProviderPool::from_env();
    if pool.providers.is_empty() {
        return Err(boxed_error("No provider available. Run `cargo run -- providers`, then enable Ollama with OLLAMA_ENABLED=true or add OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, XAI_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY or TOGETHER_API_KEY to .env."));
    }

    let (mut trainer, completed_steps) = if resume {
        match load_checkpoint(&checkpoint_path) {
            Ok(checkpoint) => {
                println!("resuming from checkpoint `{}` at completed_steps={}", checkpoint_path, checkpoint.completed_steps);
                let mut trainer = checkpoint.trainer;
                apply_memory_limits(&mut trainer);
                (trainer, checkpoint.completed_steps)
            }
            Err(error) => {
                println!("could not load checkpoint `{}`: {}", checkpoint_path, error);
                println!("starting a new training session");
                let mut trainer = new_configured_trainer();
                configure_demo_graph(&mut trainer);
                trainer.pack_state.max_items_per_theme = items_per_theme;
                trainer.training_pack.max_depth = depth;
                trainer.start_pack_training();
                (trainer, 0)
            }
        }
    } else {
        let mut trainer = new_configured_trainer();
        configure_demo_graph(&mut trainer);
        trainer.pack_state.max_items_per_theme = items_per_theme;
        trainer.training_pack.max_depth = depth;
        trainer.start_pack_training();
        (trainer, 0)
    };

    let checkpoint_has_no_work = trainer.pack_state.queue.is_empty() || trainer.pack_state.status == PackStatus::Finished;

    if resume && checkpoint_has_no_work {
        if extend || auto_extend_empty_resume {
            println!("checkpoint queue is empty; extending training with a new pack while keeping the learned state.");
            trainer.pack_state.max_items_per_theme = items_per_theme;
            trainer.training_pack.max_depth = depth;
            trainer.start_pack_training();
        } else {
            println!("checkpoint is already complete: queue=0 and status={:?}", trainer.pack_state.status);
            println!("Use `--extend` to continue with a new pack, or delete `{}` to start fresh.", checkpoint_path);
        }
    } else if trainer.pack_state.status == PackStatus::Finished && !resume {
        trainer.start_pack_training();
    }
    apply_memory_limits(&mut trainer);

    let (input_node, output_node) = demo_input_output_nodes(&trainer);

    println!("Bricks AI teacher training started.");
    print_compute_summary(&device);
    println!("provider_mode={:?}", pool.mode);
    println!("providers available={}", pool.providers.len());
    for provider in &pool.providers {
        println!(
            "- {} / {} / min_delay={}s / rpm={}",
            provider.display_name(),
            provider.model,
            provider.min_interval.as_secs(),
            provider.requests_per_minute
        );
    }
    println!("steps={steps}, items_per_theme={items_per_theme}, max_depth={depth}, checkpoint={checkpoint_path}");
    println!(
        "parallel_dimensions={} enabled={} cross_validate={} min_agreement={} max_cross_validations={}",
        dimension_config.active_dimension_count(),
        dimension_config.enabled,
        dimension_config.cross_validate,
        dimension_config.min_agreement,
        dimension_config.max_cross_validations
    );
    println!(
        "convergence enabled={} radius={} min_votes={} reinforce_supporting_paths={} final_boost={} path_boost={} correlation_boost_repeats={} correlation_score_boost={:.2} correlation_coeff_boost={:.2} survival_min_visits={} survival_bonus={:.3}",
        convergence_config.enabled,
        convergence_config.cluster_radius,
        convergence_config.min_votes,
        convergence_config.reinforce_supporting_paths,
        convergence_config.final_boost_repeats,
        convergence_config.path_boost_repeats,
        convergence_config.correlation_boost_repeats,
        convergence_config.correlation_score_boost,
        convergence_config.correlation_coefficient_boost,
        convergence_config.correlation_survival_min_visits,
        convergence_config.correlation_survival_bonus
    );
    println!(
        "parallel outputs converge={} output_radius={} max_cross_validations={}",
        dimension_config.converge_outputs,
        dimension_config.output_convergence_radius,
        dimension_config.max_cross_validations
    );
    println!(
        "pre_final_destruction enabled={} min_score={:.2} max_loss={:.2} min_alignment={:.2} protect_confidence={:.2}",
        pre_final_config.enabled,
        pre_final_config.min_candidate_score,
        pre_final_config.max_candidate_loss,
        pre_final_config.min_alignment_score,
        pre_final_config.protect_case_confidence
    );
    println!(
        "knowledge_packs enabled={} mode={} dir={} train_first={} save_generated={}",
        knowledge_config.enabled,
        knowledge_config.mode,
        knowledge_config.dir,
        knowledge_config.train_from_pack_first,
        knowledge_config.save_generated_items
    );
    println!("shortcuts: Ctrl+P or P = pause | Ctrl+S or S = resume | Ctrl+Q or Q = save checkpoint and stop");
    println!("note: pause is applied between jobs, not in the middle of a running HTTP request.\n");

    let mut benchmark = BenchmarkRecorder::new(
        steps,
        items_per_theme,
        depth,
        &checkpoint_path,
        &device,
        &pool,
        &trainer,
        dimension_config.clone(),
        convergence_config.clone(),
        pre_final_config.clone(),
        knowledge_config.clone(),
    );
    benchmark.record_event(&trainer, 0, "training_started", None, &[], 0, 0, 0, 0);

    let checkpoint_start = Instant::now();
    save_checkpoint(&checkpoint_path, completed_steps, &trainer)?;
    benchmark.add_timing("checkpoint_save", checkpoint_start.elapsed());

    let mut controls = TerminalControls::new();
    let target_trained_items = trainer.pack_state.trained_items + steps;
    let max_internal_ticks_per_step = env_usize("BRICKS_AI_MAX_INTERNAL_TICKS_PER_STEP").unwrap_or(24).max(1);
    let max_internal_ticks = steps
        .saturating_mul(max_internal_ticks_per_step)
        .max(steps + 50);
    let mut internal_ticks = 0usize;
    let progress = TrainingProgress::new(target_trained_items, &dimension_config);

    println!(
        "productive_target={} trained item(s); internal_tick_budget={}",
        steps,
        max_internal_ticks
    );
    println!("note: expansion/validation ticks no longer consume the requested training step count.\n");

    while trainer.pack_state.trained_items < target_trained_items && internal_ticks < max_internal_ticks {
        let productive_step = trainer.pack_state.trained_items;
        progress.render(&trainer, productive_step, "waiting", &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing);
        if !handle_training_controls(&mut controls, &checkpoint_path, productive_step, &trainer)? {
            break;
        }

        internal_ticks += 1;

        match trainer.pack_training_tick() {
            PackTickAction::NeedSubthemeExpansion { job, prompt } => {
                println!();
                println!(
                    "--- internal tick {internal_ticks}: expanding {:?} | trained {}/{} ---",
                    job.theme_path,
                    trainer.pack_state.trained_items,
                    target_trained_items
                );

                let expansion_started = Instant::now();
                let expansion_result = pool.generate_parsed(&prompt, |text| parse_subthemes(text).map_err(|e| e.to_string()));
                let expansion_duration = expansion_started.elapsed();
                benchmark.add_timing("subtheme_expansion", expansion_duration);

                match expansion_result {
                    Ok((provider_name, subthemes)) => {
                        let subtheme_count = subthemes.len();
                        println!("provider={} accepted_subthemes={}", provider_name, subtheme_count);
                        trainer.accept_generated_subthemes(&job, subthemes);
                        apply_memory_limits(&mut trainer);
                        benchmark.record_event_with_duration(
                            &trainer,
                            internal_ticks,
                            "subthemes_accepted",
                            Some(&provider_name),
                            &job.theme_path,
                            subtheme_count,
                            0,
                            0,
                            0,
                            Some(expansion_duration),
                        );
                        let checkpoint_started = Instant::now();
                        save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                        benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                        progress.render(&trainer, trainer.pack_state.trained_items, "subthemes accepted", &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing);
                        if !handle_training_controls(&mut controls, &checkpoint_path, trainer.pack_state.trained_items, &trainer)? {
                            break;
                        }
                    }
                    Err(error) => {
                        let failed_theme_path = job.theme_path.clone();
                        trainer.pack_state.queue.push_front(job);
                        let checkpoint_started = Instant::now();
                        save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                        benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                        println!("all providers are unavailable while expanding the theme.");
                        println!("checkpoint saved and job preserved: {error}");
                        println!("resume later with: cargo run -- train-pack --resume --steps {steps}");
                        benchmark.record_event(&trainer, internal_ticks, "provider_failure_expand", None, &failed_theme_path, 0, 0, 0, 0);
                        return Ok(());
                    }
                }
            }
            PackTickAction::NeedTrainingData { job, prompt: _prompt } => {
                println!();
                println!(
                    "--- internal tick {internal_ticks}: training data {:?} | trained {}/{} ---",
                    job.theme_path,
                    trainer.pack_state.trained_items,
                    target_trained_items
                );

                let domain = domain_from_theme_path(&job.theme_path);
                let mut knowledge_pack: Option<KnowledgePack> = None;
                let mut reused_from_pack = 0usize;
                let mut trained_from_pack = 0usize;
                let mut saved_pack_after_reuse = false;

                if !knowledge_config.continuous_only() && knowledge_config.train_from_pack_first {
                    let pack_load_started = Instant::now();
                    let (mut pack, loaded_file) = load_knowledge_pack(&knowledge_config, &domain)?;
                    benchmark.add_timing("knowledge_pack_load", pack_load_started.elapsed());
                    if !pack.is_real_enough(&knowledge_config) {
                        let build_started = Instant::now();
                        let build_outcome = ensure_real_knowledge_pack(
                            &mut pool,
                            &knowledge_config,
                            &domain,
                            pack,
                            &mut benchmark,
                        );
                        benchmark.add_timing("training_data_generation", build_started.elapsed());
                        let (mut real_pack, built_real_pack) = match build_outcome {
                            Ok(value) => value,
                            Err(error) => {
                                let failed_theme_path = job.theme_path.clone();
                                trainer.pack_state.queue.push_front(job);
                                let checkpoint_started = Instant::now();
                                save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                                benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                                println!("knowledge_pack domain={} build_failed_no_tiny_fallback", domain);
                                println!("checkpoint saved and job preserved: {error}");
                                println!("resume later with: cargo run -- train-pack --resume --steps {steps}");
                                benchmark.record_event(&trainer, internal_ticks, "knowledge_pack_build_failed", Some("knowledge_pack"), &failed_theme_path, 0, 0, 0, 0);
                                return Ok(());
                            }
                        };
                        if built_real_pack {
                            let pack_save_started = Instant::now();
                            save_knowledge_pack(&knowledge_config, &mut real_pack)?;
                            benchmark.add_timing("knowledge_pack_save", pack_save_started.elapsed());
                            benchmark.record_knowledge_pack_saved(&domain, 0, 0, true);
                        }
                        pack = real_pack;
                    }
                    let reusable = take_knowledge_items_for_job(
                        &mut pack,
                        &job,
                        job.requested_items,
                        knowledge_config.max_reuse_per_item,
                    );
                    reused_from_pack = reusable.len();
                    benchmark.record_knowledge_pack_load(&domain, reused_from_pack, loaded_file);

                    if reused_from_pack > 0 {
                        println!(
                            "knowledge_pack domain={} reused_items={} loaded_file={} total_items={} accepted_items={}",
                            domain,
                            reused_from_pack,
                            loaded_file,
                            pack.items.len(),
                            pack.accepted_count()
                        );
                        let model_training_started = Instant::now();
                        let trained_before_pack = trainer.pack_state.trained_items;
                        for validated in reusable {
                            if train_validated_item_through_dimensions(
                                &mut trainer,
                                &mut benchmark,
                                internal_ticks,
                                &job.theme_path,
                                &dimension_config,
                                &convergence_config,
                                &pre_final_config,
                                &validated,
                                "knowledge_pack",
                            ) {
                                trained_from_pack += 1;
                            }
                        }
                        benchmark.add_timing("model_training", model_training_started.elapsed());
                        let trained_delta = trainer.pack_state.trained_items.saturating_sub(trained_before_pack);
                        benchmark.record_event(
                            &trainer,
                            internal_ticks,
                            "knowledge_pack_reused",
                            Some("knowledge_pack"),
                            &job.theme_path,
                            reused_from_pack,
                            trained_from_pack,
                            0,
                            trained_delta,
                        );
                        let pack_save_started = Instant::now();
                        save_knowledge_pack(&knowledge_config, &mut pack)?;
                        benchmark.add_timing("knowledge_pack_save", pack_save_started.elapsed());
                        benchmark.record_knowledge_pack_saved(&domain, 0, 0, true);
                        saved_pack_after_reuse = true;
                    }
                    knowledge_pack = Some(pack);
                }

                let remaining_items = job.requested_items.saturating_sub(reused_from_pack);
                if remaining_items == 0 {
                    trainer.engrave_validated_weights(0.02, 0.02);
                    apply_memory_limits(&mut trainer);
                    let checkpoint_started = Instant::now();
                    save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                    benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                    progress.render(&trainer, trainer.pack_state.trained_items, "knowledge pack reused", &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing);
                    if !handle_training_controls(&mut controls, &checkpoint_path, trainer.pack_state.trained_items, &trainer)? {
                        break;
                    }
                    println!("job knowledge_pack_reused={trained_from_pack}, generated=0, saved_pack={saved_pack_after_reuse}");
                    continue;
                }

                if knowledge_config.pack_only() || (!knowledge_config.continuous_only() && !knowledge_config.allow_live_fallback) {
                    println!(
                        "knowledge_pack strict mode: missing_items={} for domain={} theme={:?}; old tiny live-QA fallback disabled.",
                        remaining_items,
                        domain,
                        job.theme_path
                    );
                    benchmark.record_knowledge_pack_starved(&domain);
                    benchmark.record_event(&trainer, internal_ticks, "knowledge_pack_strict_missing_items", Some("knowledge_pack"), &job.theme_path, 0, trained_from_pack, 0, 0);
                    let checkpoint_started = Instant::now();
                    save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                    benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                    continue;
                }

                let mut provider_job = job.clone();
                provider_job.requested_items = remaining_items;
                let provider_prompt = build_teacher_prompt(&provider_job);
                let generation_started = Instant::now();
                let items_result = pool.generate_parsed(&provider_prompt, |text| parse_teacher_items(text).map_err(|e| e.to_string()));
                let generation_duration = generation_started.elapsed();
                benchmark.add_timing("training_data_generation", generation_duration);
                let (generator_name, items) = match items_result {
                    Ok(value) => value,
                    Err(error) => {
                        let failed_theme_path = job.theme_path.clone();
                        trainer.pack_state.queue.push_front(job);
                        let checkpoint_started = Instant::now();
                        save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                        benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                        println!("all providers are unavailable while generating training data.");
                        println!("checkpoint saved and job preserved: {error}");
                        println!("resume later with: cargo run -- train-pack --resume --steps {steps}");
                        benchmark.record_event(&trainer, internal_ticks, "provider_failure_generate", None, &failed_theme_path, 0, 0, 0, 0);
                        return Ok(());
                    }
                };

                let generated_candidate_count = items.len();
                println!(
                    "provider={} generated_candidate_items={} knowledge_pack_reused={} requested_missing={}",
                    generator_name,
                    generated_candidate_count,
                    reused_from_pack,
                    remaining_items
                );
                benchmark.record_event_with_duration(
                    &trainer,
                    internal_ticks,
                    "training_data_generated",
                    Some(&generator_name),
                    &job.theme_path,
                    generated_candidate_count,
                    0,
                    0,
                    0,
                    Some(generation_duration),
                );

                let validation_started = Instant::now();
                let validation_result = validate_items_with_pool(&mut pool, &provider_job, &items);
                let validation_duration = validation_started.elapsed();
                benchmark.add_timing("validation", validation_duration);
                let (validator_name, validation_scores) = match validation_result {
                    Ok(value) => value,
                    Err(error) => {
                        let failed_theme_path = job.theme_path.clone();
                        trainer.pack_state.queue.push_front(job);
                        let checkpoint_started = Instant::now();
                        save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                        benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                        println!("all providers are unavailable while validating training data.");
                        println!("checkpoint saved and job preserved: {error}");
                        println!("resume later with: cargo run -- train-pack --resume --steps {steps}");
                        benchmark.record_event(&trainer, internal_ticks, "provider_failure_validate", None, &failed_theme_path, 0, 0, 0, 0);
                        return Ok(());
                    }
                };

                println!("validator_provider={}", validator_name);

                if knowledge_config.enabled && knowledge_config.save_generated_items && !knowledge_config.continuous_only() {
                    let mut pack = match knowledge_pack.take() {
                        Some(pack) => pack,
                        None => {
                            let pack_load_started = Instant::now();
                            let (pack, loaded_file) = load_knowledge_pack(&knowledge_config, &domain)?;
                            benchmark.add_timing("knowledge_pack_load", pack_load_started.elapsed());
                            benchmark.record_knowledge_pack_load(&domain, 0, loaded_file);
                            let (pack, _) = ensure_real_knowledge_pack(
                                &mut pool,
                                &knowledge_config,
                                &domain,
                                pack,
                                &mut benchmark,
                            )?;
                            pack
                        }
                    };
                    let (accepted_saved, rejected_saved) = append_generated_items_to_knowledge_pack(
                        &mut pack,
                        &provider_job,
                        &items,
                        &validation_scores,
                        &generator_name,
                        &validator_name,
                        &trainer,
                    );
                    let pack_save_started = Instant::now();
                    save_knowledge_pack(&knowledge_config, &mut pack)?;
                    benchmark.add_timing("knowledge_pack_save", pack_save_started.elapsed());
                    benchmark.record_knowledge_pack_saved(&domain, accepted_saved, rejected_saved, true);
                    println!(
                        "knowledge_pack domain={} saved_accepted={} saved_rejected={} total_items={} path={}",
                        domain,
                        accepted_saved,
                        rejected_saved,
                        pack.items.len(),
                        knowledge_pack_path(&knowledge_config, &domain).display()
                    );
                }

                let mut accepted_in_job = trained_from_pack;
                let mut rejected_in_job = 0usize;
                let trained_before_job = trainer.pack_state.trained_items;

                let model_training_started = Instant::now();
                for (index, item) in items.into_iter().enumerate() {
                    let validator_score = validation_scores.get(&index).copied().unwrap_or(0.0);
                    let validated = trainer.validate_pack_item(&provider_job, item.into_core(), validator_score);

                    if validated.accepted {
                        if train_validated_item_through_dimensions(
                            &mut trainer,
                            &mut benchmark,
                            internal_ticks,
                            &job.theme_path,
                            &dimension_config,
                            &convergence_config,
                            &pre_final_config,
                            &validated,
                            &generator_name,
                        ) {
                            accepted_in_job += 1;
                        }
                    } else {
                        rejected_in_job += 1;
                        trainer.pack_state.rejected_items += 1;
                    }
                }
                let model_training_duration = model_training_started.elapsed();
                benchmark.add_timing("model_training", model_training_duration);

                trainer.engrave_validated_weights(0.02, 0.02);
                apply_memory_limits(&mut trainer);
                let trained_delta = trainer.pack_state.trained_items.saturating_sub(trained_before_job);
                benchmark.record_event(
                    &trainer,
                    internal_ticks,
                    "training_data_processed",
                    Some(&validator_name),
                    &job.theme_path,
                    generated_candidate_count,
                    accepted_in_job,
                    rejected_in_job,
                    trained_delta,
                );
                let checkpoint_started = Instant::now();
                save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                progress.render(&trainer, trainer.pack_state.trained_items, "training data processed", &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing);
                if !handle_training_controls(&mut controls, &checkpoint_path, trainer.pack_state.trained_items, &trainer)? {
                    break;
                }

                println!(
                    "job accepted={} rejected={} reused_from_pack={} generated={}",
                    accepted_in_job,
                    rejected_in_job,
                    reused_from_pack,
                    generated_candidate_count
                );
            }
            PackTickAction::Finished => {
                if trainer.pack_state.trained_items < target_trained_items && (extend || auto_extend_empty_resume) {
                    println!();
                    println!(
                        "Pack finished before target trained items ({}/{}). Extending locally with a new pack while keeping learned state.",
                        trainer.pack_state.trained_items,
                        target_trained_items
                    );
                    trainer.pack_state.max_items_per_theme = items_per_theme;
                    trainer.training_pack.max_depth = depth;
                    trainer.start_pack_training();
                    benchmark.record_event(&trainer, internal_ticks, "pack_extended", None, &[], 0, 0, 0, 0);
                    let checkpoint_started = Instant::now();
                    save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                    benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                    continue;
                }

                println!("Pack finished.");
                benchmark.record_event(&trainer, internal_ticks, "pack_finished", None, &[], 0, 0, 0, 0);
                let checkpoint_started = Instant::now();
                save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                progress.render(&trainer, trainer.pack_state.trained_items, "finished", &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing);
                break;
            }
            PackTickAction::Idle => {
                println!("Pack is idle or paused.");
                benchmark.record_event(&trainer, internal_ticks, "pack_idle_or_paused", None, &[], 0, 0, 0, 0);
                let checkpoint_started = Instant::now();
                save_checkpoint(&checkpoint_path, trainer.pack_state.trained_items, &trainer)?;
                benchmark.add_timing("checkpoint_save", checkpoint_started.elapsed());
                break;
            }
        }
    }

    if trainer.pack_state.trained_items < target_trained_items {
        println!(
            "warning: stopped before requested trained item target: trained_items={} target={} internal_ticks={} budget={}",
            trainer.pack_state.trained_items,
            target_trained_items,
            internal_ticks,
            max_internal_ticks
        );
        println!("increase BRICKS_AI_MAX_INTERNAL_TICKS_PER_STEP or reduce --depth if the model spends too long expanding themes.");
    }

    let prediction = trainer.predict(&[(input_node, 1.0)], &[output_node], 1);
    let engraved = trainer.export_engraved_model();

    println!();
    println!("training summary:");
    println!("accepted_items={}", trainer.pack_state.accepted_items);
    println!("rejected_items={}", trainer.pack_state.rejected_items);
    println!("trained_items={}", trainer.pack_state.trained_items);
    println!("queue_remaining={}", trainer.pack_state.queue.len());
    println!("engraved_cases={}", engraved.cases.len());
    println!("engraved_correlations={}", engraved.correlations.len());
    println!("final_probe_prediction={:.4}", prediction[0]);

    let path = "engraved_model.json";
    trainer.save_engraved_model(path)?;
    benchmark.record_event(&trainer, internal_ticks, "training_completed", None, &[], 0, 0, 0, 0);
    let cutoff_reason = if controls.stop_requested {
        "user_stop"
    } else if trainer.pack_state.trained_items >= target_trained_items {
        "target_reached"
    } else if internal_ticks >= max_internal_ticks {
        "internal_tick_budget_exhausted"
    } else if trainer.pack_state.status == PackStatus::Finished {
        "pack_finished"
    } else {
        "loop_stopped"
    };
    progress.finish(&trainer, &controls, &benchmark.report.dimension_stats, &benchmark.report.convergence_stats, &benchmark.report.timing, cutoff_reason);
    benchmark.finish(
        &trainer,
        &device,
        internal_ticks,
        target_trained_items,
        prediction[0],
        path,
        &checkpoint_path,
        controls.total_paused_ms(),
        controls.stop_requested,
        cutoff_reason,
    );
    let (benchmark_json_path, benchmark_csv_path) = write_benchmark_files(&benchmark.report)?;

    println!("saved {path}");
    println!("checkpoint saved {checkpoint_path}");
    print_benchmark_report(&benchmark.report, &benchmark_json_path, &benchmark_csv_path);

    Ok(())
}


struct TerminalControls {
    raw_mode_enabled: bool,
    paused: bool,
    stop_requested: bool,
    pause_started_at: Option<Instant>,
    total_paused: Duration,
}

impl TerminalControls {
    fn new() -> Self {
        let raw_mode_enabled = enable_raw_mode().is_ok();
        if !raw_mode_enabled {
            println!("keyboard shortcuts disabled: terminal raw mode is unavailable");
        }

        Self {
            raw_mode_enabled,
            paused: false,
            stop_requested: false,
            pause_started_at: None,
            total_paused: Duration::ZERO,
        }
    }

    fn start_pause(&mut self) {
        if !self.paused {
            self.paused = true;
            self.pause_started_at = Some(Instant::now());
            println!("\npaused. Press Ctrl+S or S to resume. Press Ctrl+Q or Q to save and stop.");
        }
    }

    fn resume(&mut self) {
        if self.paused {
            if let Some(start) = self.pause_started_at.take() {
                self.total_paused += start.elapsed();
            }
            self.paused = false;
            println!("\nresumed.");
        }
    }

    fn request_stop(&mut self) {
        if self.paused {
            if let Some(start) = self.pause_started_at.take() {
                self.total_paused += start.elapsed();
            }
        }
        self.stop_requested = true;
        println!("\nstop requested. Saving checkpoint...");
    }

    fn total_paused_ms(&self) -> u64 {
        let mut total = self.total_paused;
        if self.paused {
            if let Some(start) = self.pause_started_at {
                total += start.elapsed();
            }
        }
        millis_u64(total)
    }

    fn poll(&mut self) {
        if !self.raw_mode_enabled {
            return;
        }

        while let Ok(true) = event::poll(Duration::from_millis(0)) {
            match event::read() {
                Ok(Event::Key(key)) => {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match (ctrl, key.code) {
                        (true, KeyCode::Char('p')) | (true, KeyCode::Char('P')) => self.start_pause(),
                        (true, KeyCode::Char('s')) | (true, KeyCode::Char('S')) => self.resume(),
                        (true, KeyCode::Char('q')) | (true, KeyCode::Char('Q')) => self.request_stop(),
                        (false, KeyCode::Char('p')) | (false, KeyCode::Char('P')) => self.start_pause(),
                        (false, KeyCode::Char('s')) | (false, KeyCode::Char('S')) => self.resume(),
                        (false, KeyCode::Char('q')) | (false, KeyCode::Char('Q')) => self.request_stop(),
                        _ => {}
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
}

impl Drop for TerminalControls {
    fn drop(&mut self) {
        if self.raw_mode_enabled {
            let _ = disable_raw_mode();
        }
    }
}

fn handle_training_controls(
    controls: &mut TerminalControls,
    checkpoint_path: &str,
    checkpoint_step: usize,
    trainer: &RawAITrainer,
) -> Result<bool, Box<dyn Error>> {
    controls.poll();

    if controls.stop_requested {
        save_checkpoint(checkpoint_path, checkpoint_step, trainer)?;
        println!("checkpoint saved {checkpoint_path}");
        return Ok(false);
    }

    if controls.paused {
        save_checkpoint(checkpoint_path, checkpoint_step, trainer)?;
        println!("checkpoint saved {checkpoint_path}");
        println!("training paused. Shortcuts: Ctrl+S/S resume | Ctrl+Q/Q save and stop");

        while controls.paused && !controls.stop_requested {
            thread::sleep(Duration::from_millis(200));
            controls.poll();
        }

        if controls.stop_requested {
            save_checkpoint(checkpoint_path, checkpoint_step, trainer)?;
            println!("checkpoint saved {checkpoint_path}");
            return Ok(false);
        }
    }

    Ok(true)
}

struct TrainingProgress {
    bar: ProgressBar,
    target: usize,
}

impl TrainingProgress {
    fn new(target: usize, dimensions: &DimensionConfig) -> Self {
        let bar = ProgressBar::new(target.max(1) as u64);
        let style = ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:34.cyan/blue}] {percent:>3}% {pos}/{len} ETA:{eta_precise} {msg}"
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("=>-");
        bar.set_style(style);
        bar.set_message(format!(
            "queue=0 trained=0 accepted=0 rejected=0 dims={} status=starting",
            dimensions.active_dimension_count()
        ));
        Self { bar, target: target.max(1) }
    }

    fn render(
        &self,
        trainer: &RawAITrainer,
        current_step: usize,
        status: &str,
        controls: &TerminalControls,
        dimensions: &BenchmarkDimensionStats,
        convergence: &BenchmarkConvergenceStats,
        timing: &BenchmarkTimingTotals,
    ) {
        let finished = trainer.pack_state.queue.is_empty() || trainer.pack_state.status == PackStatus::Finished;
        let display_step = if finished { self.target } else { current_step.min(self.target) };
        self.bar.set_position(display_step as u64);
        self.bar.set_message(format!(
            "queue={} trained={} accepted={} rejected={} paths={} dims={} winners={} conv={:.3} sleep={} pause={} status={}",
            trainer.pack_state.queue.len(),
            trainer.pack_state.trained_items,
            trainer.pack_state.accepted_items,
            trainer.pack_state.rejected_items,
            trainer.paths.iter().filter(|path| path.active).count(),
            dimensions.parallel_dimensions,
            dimensions.unique_winner_paths,
            convergence.max_winner_score,
            format_duration_ms(timing.sleep_interruption_ms),
            format_duration_ms(controls.total_paused_ms()),
            status,
        ));
        self.bar.tick();
    }

    fn finish(
        &self,
        trainer: &RawAITrainer,
        controls: &TerminalControls,
        dimensions: &BenchmarkDimensionStats,
        convergence: &BenchmarkConvergenceStats,
        timing: &BenchmarkTimingTotals,
        status: &str,
    ) {
        self.bar.set_position(self.target as u64);
        self.bar.finish_with_message(format!(
            "trained={} accepted={} rejected={} paths={} dims={} winners={} conv={:.3} sleep={} pause={} status={}",
            trainer.pack_state.trained_items,
            trainer.pack_state.accepted_items,
            trainer.pack_state.rejected_items,
            trainer.paths.iter().filter(|path| path.active).count(),
            dimensions.parallel_dimensions,
            dimensions.unique_winner_paths,
            convergence.max_winner_score,
            format_duration_ms(timing.sleep_interruption_ms),
            format_duration_ms(controls.total_paused_ms()),
            status,
        ));
    }
}


#[derive(Debug, Clone)]
struct ConvergenceClusterTemp {
    cluster_id: usize,
    representative: NodeId,
    member_indices: Vec<usize>,
    neighbor_merges: usize,
    convergence_score: f32,
}

fn pre_final_alignment_score(
    trainer: &RawAITrainer,
    candidate: &DimensionCandidateScore,
) -> f32 {
    let output_grid = trainer.grids.len().saturating_sub(1);
    let input_alignment = if candidate.path.input.grid == 0 { 1.0 } else { 0.0 };
    let output_alignment = if candidate.path.output.grid == output_grid { 1.0 } else { 0.0 };
    let loss_reward = 1.0 / (1.0 + candidate.loss.max(0.0));
    ((0.42 * candidate.score.clamp(0.0, 1.0))
        + (0.34 * loss_reward.clamp(0.0, 1.0))
        + (0.12 * input_alignment)
        + (0.12 * output_alignment))
        .clamp(0.0, 1.0)
}

fn pre_final_destruction_reason(
    trainer: &RawAITrainer,
    candidate: &DimensionCandidateScore,
    config: &PreFinalDestructionConfig,
) -> Option<String> {
    let output_grid = trainer.grids.len().saturating_sub(1);
    if candidate.path.input.grid != 0 {
        return Some("not_aligned_with_initial_input_block".to_string());
    }
    if candidate.path.output.grid != output_grid {
        return Some("not_aligned_with_final_output_block".to_string());
    }
    if candidate.loss > config.max_candidate_loss {
        return Some(format!("loss_above_prefinal_limit:{:.6}", candidate.loss));
    }
    if candidate.score < config.min_candidate_score {
        return Some(format!("candidate_score_below_prefinal_limit:{:.6}", candidate.score));
    }
    let alignment = pre_final_alignment_score(trainer, candidate);
    if alignment < config.min_alignment_score {
        return Some(format!("input_output_alignment_below_limit:{:.6}", alignment));
    }
    None
}

fn candidate_nodes_for_prefinal_destruction(
    candidate: &DimensionCandidateScore,
    config: &PreFinalDestructionConfig,
) -> Vec<NodeId> {
    let mut nodes = vec![candidate.path.input];
    if config.destroy_middle_nodes {
        nodes.push(candidate.path.mid_1);
        nodes.push(candidate.path.mid_2);
    }
    if config.destroy_output_nodes {
        nodes.push(candidate.path.output);
    }
    nodes
}

fn apply_pre_final_destruction(
    trainer: &mut RawAITrainer,
    candidates: Vec<DimensionCandidateScore>,
    config: &PreFinalDestructionConfig,
) -> (Vec<DimensionCandidateScore>, PreFinalDestructionRunStats) {
    let mut stats = PreFinalDestructionRunStats {
        candidates_seen: candidates.len(),
        ..PreFinalDestructionRunStats::default()
    };

    if !config.enabled || candidates.len() <= 1 {
        stats.candidates_forwarded = candidates.len();
        return (candidates, stats);
    }

    trainer.pre_final_destruction.enabled = true;
    trainer.pre_final_destruction.runs += 1;
    trainer.pre_final_destruction.candidates_seen += candidates.len();

    let best_index = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.loss.partial_cmp(&a.loss).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);

    let mut forwarded = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.into_iter().enumerate() {
        let alignment = pre_final_alignment_score(trainer, &candidate);
        stats.last_alignment_score = alignment;
        let reason = pre_final_destruction_reason(trainer, &candidate, config);
        let keep_as_best_rescue = config.always_keep_best_candidate && index == best_index;

        if let Some(reason) = reason {
            if keep_as_best_rescue {
                stats.candidates_rescued += 1;
                trainer.pre_final_destruction.candidates_rescued += 1;
                forwarded.push(candidate);
            } else {
                let path_key = candidate.path.key();
                let nodes = candidate_nodes_for_prefinal_destruction(&candidate, config);
                let (cases_destroyed, correlations_destroyed) = trainer.destroy_prefinal_candidate_path(
                    &path_key,
                    &nodes,
                    &reason,
                    config.protect_case_confidence,
                );
                stats.candidates_destroyed += 1;
                stats.blocks_destroyed += 1;
                stats.cases_destroyed += cases_destroyed;
                stats.correlations_destroyed += correlations_destroyed;
                stats.last_reason = Some(reason);
            }
        } else {
            forwarded.push(candidate);
        }
    }

    if forwarded.is_empty() {
        // This should not happen with always_keep_best_candidate, but keep the trainer safe.
        stats.candidates_rescued += 1;
    }

    stats.candidates_forwarded = forwarded.len();
    trainer.pre_final_destruction.candidates_forwarded += forwarded.len();
    (forwarded, stats)
}

fn train_with_parallel_dimensions(
    trainer: &mut RawAITrainer,
    item: &ValidatedPackItem,
    config: &DimensionConfig,
    convergence: &ConvergenceConfig,
    pre_final: &PreFinalDestructionConfig,
) -> Option<DimensionTrainingResult> {
    if !item.accepted {
        trainer.pack_state.rejected_items += 1;
        return None;
    }

    let dimension_count = config.active_dimension_count();
    trainer.ensure_parallel_dimensions(dimension_count);
    let mut created_paths = 0usize;
    let mut candidates = Vec::with_capacity(dimension_count);

    for dimension_id in 0..dimension_count {
        let path = build_dimension_path(trainer, item, dimension_id, config);
        created_paths += ensure_dimension_path(trainer, &path);

        let mut probe = trainer.clone_for_checkpoint();
        let input_value = hash_text_to_signal(&item.question);
        let expected_value = hash_text_to_signal(&item.answer);
        let loss = probe.train_step(&[(path.input, input_value)], &[(path.output, expected_value)], 1);
        let reward = 1.0 / (1.0 + loss);
        let teacher = item.teacher_confidence.clamp(0.0, 1.0);
        let validator = item.validator_score.clamp(0.0, 1.0);
        let dimension_diversity = if dimension_id == 0 { 0.0 } else { config.diversity_bonus };
        let score = (0.60 * reward) + (0.25 * validator) + (0.10 * teacher) + (0.05 * dimension_diversity);
        candidates.push(DimensionCandidateScore {
            path,
            loss,
            score,
            convergence_score: 0.0,
            cross_validated: false,
        });
    }

    if candidates.is_empty() {
        return None;
    }

    let (mut candidates, pre_final_stats) = apply_pre_final_destruction(trainer, candidates, pre_final);
    if candidates.is_empty() {
        return None;
    }

    let mut clusters = build_convergence_clusters(&candidates, convergence);
    score_convergence_clusters(&mut clusters, &candidates, convergence, dimension_count);
    clusters.sort_by(|a, b| {
        b.convergence_score
            .partial_cmp(&a.convergence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.member_indices.len().cmp(&a.member_indices.len()))
    });

    let fallback_index = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.loss.partial_cmp(&a.loss).unwrap_or(std::cmp::Ordering::Equal))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);

    let winner_cluster = clusters
        .iter()
        .find(|cluster| !convergence.enabled || cluster.member_indices.len() >= convergence.min_votes)
        .or_else(|| {
            if convergence.require_cross_validation {
                None
            } else {
                clusters.first()
            }
        })
        .cloned()
        .unwrap_or_else(|| ConvergenceClusterTemp {
            cluster_id: 0,
            representative: candidates[fallback_index].path.output,
            member_indices: vec![fallback_index],
            neighbor_merges: 0,
            convergence_score: candidates[fallback_index].score,
        });

    let winner_index = winner_cluster
        .member_indices
        .iter()
        .copied()
        .max_by(|a, b| {
            candidates[*a]
                .score
                .partial_cmp(&candidates[*b].score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| candidates[*b].loss.partial_cmp(&candidates[*a].loss).unwrap_or(std::cmp::Ordering::Equal))
        })
        .unwrap_or(fallback_index);

    let winner = candidates[winner_index].clone();
    let agreement_count = candidates
        .iter()
        .filter(|candidate| candidate.loss <= winner.loss + config.loss_agreement_margin)
        .count()
        .max(1);

    let input_value = hash_text_to_signal(&item.question);
    let expected_value = hash_text_to_signal(&item.answer);
    let training_start = Instant::now();
    let winner_loss = trainer.train_step(&[(winner.path.input, input_value)], &[(winner.path.output, expected_value)], 1);
    let mut supporting_paths_reinforced = 0usize;
    let mut cross_validated_paths = 0usize;
    let final_node = winner.path.output;

    if config.cross_validate && agreement_count >= config.min_agreement {
        for index in winner_cluster.member_indices.iter().copied() {
            if index == winner_index {
                candidates[index].cross_validated = true;
                continue;
            }
            if cross_validated_paths >= config.max_cross_validations {
                break;
            }
            if candidates[index].loss <= winner.loss + config.loss_agreement_margin {
                let candidate_path = candidates[index].path.clone();
                if convergence.reinforce_supporting_paths {
                    ensure_convergence_bridge(trainer, &candidate_path, final_node);
                    let _ = trainer.train_step(&[(candidate_path.input, input_value)], &[(final_node, expected_value)], 1);
                    supporting_paths_reinforced += 1;
                } else {
                    let _ = trainer.train_step(&[(candidate_path.input, input_value)], &[(candidate_path.output, expected_value)], 1);
                }
                candidates[index].cross_validated = true;
                cross_validated_paths += 1;
            }
        }
    }

    let convergence_reward = winner_cluster.convergence_score.clamp(0.0, 1.0);
    let final_repeats = winner_cluster
        .member_indices
        .len()
        .max(1)
        .saturating_mul(convergence.final_boost_repeats.max(1));
    trainer.reinforce_node_confidence(final_node, convergence_reward, final_repeats);
    let path_reward = (convergence_reward * 0.55).clamp(0.0, 1.0);
    let correlation_repeats = convergence
        .correlation_boost_repeats
        .max(1)
        .saturating_add(winner_cluster.member_indices.len().saturating_sub(1) / 4);
    for index in &winner_cluster.member_indices {
        let candidate_path = candidates[*index].path.nodes();
        let candidate_reward = if *index == winner_index {
            convergence_reward
        } else {
            path_reward
        };
        for node in candidate_path {
            trainer.reinforce_node_confidence(node, path_reward, convergence.path_boost_repeats.max(1));
        }
        trainer.reinforce_path_correlations(
            &candidate_path,
            candidate_reward,
            correlation_repeats,
            convergence.correlation_score_boost,
            convergence.correlation_coefficient_boost,
            convergence.correlation_survival_min_visits,
            convergence.correlation_survival_bonus,
        );
    }
    trainer.pack_state.accepted_items += 1;
    trainer.pack_state.trained_items += 1;

    let singleton_rejections = clusters.iter().filter(|cluster| cluster.member_indices.len() == 1).count();
    for index in &winner_cluster.member_indices {
        candidates[*index].convergence_score = winner_cluster.convergence_score;
    }

    let race_records: Vec<DimensionRaceCandidate> = candidates
        .iter()
        .map(|candidate| DimensionRaceCandidate {
            dimension_id: candidate.path.dimension_id,
            path_key: candidate.path.key(),
            input: candidate.path.input,
            mid_1: candidate.path.mid_1,
            mid_2: candidate.path.mid_2,
            output: candidate.path.output,
            loss: candidate.loss,
            score: candidate.score,
            agreement_count,
            cross_validated: candidate.cross_validated,
            convergence_score: candidate.convergence_score,
        })
        .collect();

    let mut ordered_member_indices = vec![winner_index];
    for index in &winner_cluster.member_indices {
        if *index != winner_index {
            ordered_member_indices.push(*index);
        }
    }
    let contributing_dimensions: Vec<usize> = ordered_member_indices
        .iter()
        .map(|index| candidates[*index].path.dimension_id)
        .collect();
    let contributing_paths: Vec<String> = ordered_member_indices
        .iter()
        .map(|index| candidates[*index].path.key())
        .collect();
    let member_cases: Vec<usize> = winner_cluster
        .member_indices
        .iter()
        .map(|index| candidates[*index].path.output.case)
        .collect();
    let avg_loss = average_f32(winner_cluster.member_indices.iter().map(|index| candidates[*index].loss));
    let avg_score = average_f32(winner_cluster.member_indices.iter().map(|index| candidates[*index].score));
    let cluster_record = ConvergenceClusterRecord {
        cluster_id: winner_cluster.cluster_id,
        final_node,
        representative_case: final_node.case,
        member_cases,
        candidate_count: winner_cluster.member_indices.len(),
        vote_weight: winner_cluster.member_indices.len() as f32 / dimension_count.max(1) as f32,
        avg_loss,
        avg_candidate_score: avg_score,
        convergence_score: winner_cluster.convergence_score,
        contributing_dimensions,
        contributing_paths,
        neighbor_merges: winner_cluster.neighbor_merges,
        supporting_paths_reinforced,
    };
    trainer.record_dimension_convergence(dimension_count, &race_records, cluster_record);

    let _training_elapsed = training_start.elapsed();

    Some(DimensionTrainingResult {
        winner_dimension_id: winner.path.dimension_id,
        winner_loss,
        winner_score: winner.score,
        agreement_count,
        cross_validated_paths,
        candidate_count: candidates.len(),
        pre_final_stats,
        winner_path_key: winner.path.key(),
        created_paths,
        convergence_score: winner_cluster.convergence_score,
        convergence_votes: winner_cluster.member_indices.len(),
        convergence_final_node: final_node,
        convergence_neighbor_merges: winner_cluster.neighbor_merges,
        convergence_supporting_paths_reinforced: supporting_paths_reinforced,
        convergence_singleton_rejections: singleton_rejections,
    })
}

fn build_convergence_clusters(
    candidates: &[DimensionCandidateScore],
    config: &ConvergenceConfig,
) -> Vec<ConvergenceClusterTemp> {
    if !config.enabled {
        return candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| ConvergenceClusterTemp {
                cluster_id: index,
                representative: candidate.path.output,
                member_indices: vec![index],
                neighbor_merges: 0,
                convergence_score: candidate.score,
            })
            .collect();
    }

    let mut clusters: Vec<ConvergenceClusterTemp> = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let output = candidate.path.output;
        let mut merged = false;
        for cluster in &mut clusters {
            if cluster.representative.grid == output.grid
                && cluster.representative.page == output.page
                && cluster.representative.case.abs_diff(output.case) <= config.cluster_radius
            {
                cluster.member_indices.push(index);
                if cluster.representative.case != output.case {
                    cluster.neighbor_merges += 1;
                }
                merged = true;
                break;
            }
        }
        if !merged {
            let cluster_id = clusters.len();
            clusters.push(ConvergenceClusterTemp {
                cluster_id,
                representative: output,
                member_indices: vec![index],
                neighbor_merges: 0,
                convergence_score: 0.0,
            });
        }
    }
    clusters
}

fn score_convergence_clusters(
    clusters: &mut [ConvergenceClusterTemp],
    candidates: &[DimensionCandidateScore],
    config: &ConvergenceConfig,
    dimension_count: usize,
) {
    let total_weight = (config.weight_votes
        + config.weight_candidate_score
        + config.weight_loss_reward
        + config.weight_stability)
        .max(0.0001);
    for cluster in clusters {
        let votes = cluster.member_indices.len();
        let vote_score = votes as f32 / dimension_count.max(1) as f32;
        let avg_candidate_score = average_f32(cluster.member_indices.iter().map(|index| candidates[*index].score));
        let avg_loss = average_f32(cluster.member_indices.iter().map(|index| candidates[*index].loss));
        let loss_reward = 1.0 / (1.0 + avg_loss);
        let stability = if votes >= config.min_votes { 1.0 } else { votes as f32 / config.min_votes.max(1) as f32 };
        cluster.convergence_score = ((config.weight_votes * vote_score)
            + (config.weight_candidate_score * avg_candidate_score)
            + (config.weight_loss_reward * loss_reward)
            + (config.weight_stability * stability))
            / total_weight;
    }
}

fn ensure_convergence_bridge(trainer: &mut RawAITrainer, path: &DimensionPath, final_node: NodeId) {
    if path.output != final_node && !correlation_exists(trainer, path.mid_2, final_node) {
        trainer.add_correlation(path.mid_2, final_node, 0.20);
        trainer.discover_paths_from(path.input, 4, 12);
    }
}

fn build_dimension_path(
    trainer: &RawAITrainer,
    item: &ValidatedPackItem,
    dimension_id: usize,
    config: &DimensionConfig,
) -> DimensionPath {
    let grid_count = trainer.grids.len().max(1);
    let input_grid = 0usize;
    let mid_1_grid = if grid_count > 1 { 1 } else { 0 };
    let mid_2_grid = if grid_count > 2 { 2 } else { mid_1_grid };
    let output_grid = grid_count.saturating_sub(1);
    let theme = item.theme_path.join("/");
    let branch_base = stable_hash64(&format!(
        "{}|{}|{}|{}|{}|{}",
        theme,
        item.question,
        item.answer,
        dimension_id,
        config.selection,
        config.active_dimension_count(),
    ));
    let shared_output_base = stable_hash64(&format!(
        "{}|{}|{}|{}|{}|final-convergence-cube",
        theme,
        item.question,
        item.answer,
        config.selection,
        config.active_dimension_count(),
    ));

    let input = trainer.official_node(input_grid, dimension_case_index(trainer, input_grid, branch_base, 0));
    let mid_1 = trainer.official_node(mid_1_grid, dimension_case_index(trainer, mid_1_grid, rotate_mix(branch_base, 17), 1));
    let mid_2 = trainer.official_node(mid_2_grid, dimension_case_index(trainer, mid_2_grid, rotate_mix(branch_base, 31), 2));
    let output_case = if config.converge_outputs {
        converged_output_case_index(
            trainer,
            output_grid,
            rotate_mix(shared_output_base, 47),
            rotate_mix(branch_base, 59),
            config.output_convergence_radius,
        )
    } else {
        dimension_case_index(trainer, output_grid, rotate_mix(branch_base, 47), 3)
    };
    let output = trainer.official_node(output_grid, output_case);

    DimensionPath { dimension_id, input, mid_1, mid_2, output }
}

fn converged_output_case_index(
    trainer: &RawAITrainer,
    grid: usize,
    shared_hash: u64,
    branch_hash: u64,
    radius: usize,
) -> usize {
    let page = trainer.grids[grid].official_page;
    let len = trainer.grids[grid].pages[page].cases.len().max(1);
    let center = (shared_hash as usize) % len;
    let span = radius.saturating_mul(2).saturating_add(1).max(1);
    let raw_offset = (branch_hash as usize) % span;
    let signed_offset = raw_offset as isize - radius as isize;
    wrap_case_index(center, signed_offset, len)
}

fn wrap_case_index(center: usize, offset: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let len_i = len as isize;
    let wrapped = (center as isize + offset).rem_euclid(len_i);
    wrapped as usize
}

fn dimension_case_index(trainer: &RawAITrainer, grid: usize, hash: u64, salt: u64) -> usize {
    let page = trainer.grids[grid].official_page;
    let len = trainer.grids[grid].pages[page].cases.len().max(1);
    let mixed = hash ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (mixed as usize) % len
}

fn ensure_dimension_path(trainer: &mut RawAITrainer, path: &DimensionPath) -> usize {
    let nodes = path.nodes();
    let mut created = 0usize;
    for pair in nodes.windows(2) {
        if !correlation_exists(trainer, pair[0], pair[1]) {
            trainer.add_correlation(pair[0], pair[1], 0.20);
            created += 1;
        }
    }
    trainer.discover_paths_from(path.input, 4, 12);
    if created > 0 { 1 } else { 0 }
}

fn correlation_exists(trainer: &RawAITrainer, from: NodeId, to: NodeId) -> bool {
    trainer
        .correlations
        .iter()
        .any(|correlation| correlation.from == from && correlation.to == to && correlation.active)
}

fn rotate_mix(value: u64, rotate: u32) -> u64 {
    value
        .rotate_left(rotate)
        .wrapping_mul(0xD6E8_FD9D_9B4C_2D1D)
        ^ 0xA076_1D64_78BD_642F
}

fn stable_hash64(text: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn average_ms_per_item(elapsed_ms: u64, items: usize) -> f32 {
    if items == 0 { 0.0 } else { elapsed_ms as f32 / items as f32 }
}

fn format_duration_ms(ms: u64) -> String {
    let seconds = ms / 1000;
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}", m, s)
    }
}


#[derive(Debug, Clone, Copy)]
enum MemoryMode {
    Low,
    Balanced,
    Performance,
    Heavy,
}

impl MemoryMode {
    fn from_env() -> Self {
        match env::var("BRICKS_AI_MEMORY_MODE")
            .unwrap_or_else(|_| "balanced".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "low" | "minimal" | "cpu-low" => Self::Low,
            "performance" | "perf" | "fast" => Self::Performance,
            "heavy" | "gpu-heavy" | "full" => Self::Heavy,
            _ => Self::Balanced,
        }
    }

    fn defaults(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Low => (4, 256, 64, 64),
            Self::Balanced => (4, 1000, 256, 256),
            Self::Performance => (6, 2048, 1024, 1024),
            Self::Heavy => (16, 4096, 8192, 8192),
        }
    }
}

fn new_configured_trainer() -> RawAITrainer {
    let mode = MemoryMode::from_env();
    let (default_grids, default_cases, _, _) = mode.defaults();
    let grid_count = env_usize("BRICKS_AI_GRID_COUNT").unwrap_or(default_grids).max(4);
    let cases_per_grid = env_usize("BRICKS_AI_CASES_PER_GRID").unwrap_or(default_cases).max(1);
    let mut trainer = RawAITrainer::new(grid_count, cases_per_grid);
    apply_memory_limits(&mut trainer);
    trainer
}

fn apply_memory_limits(trainer: &mut RawAITrainer) {
    let mode = MemoryMode::from_env();
    let (_, _, default_queue, default_candidates) = mode.defaults();
    let max_queue = env_usize("BRICKS_AI_MAX_QUEUE_JOBS").unwrap_or(default_queue);
    let max_candidates = env_usize("BRICKS_AI_MAX_CANDIDATE_CACHE").unwrap_or(default_candidates);
    let max_paths = env_usize("BRICKS_AI_MAX_PATH_CACHE").unwrap_or(default_candidates);

    while max_queue > 0 && trainer.pack_state.queue.len() > max_queue {
        trainer.pack_state.queue.pop_back();
    }

    if max_candidates > 0 && trainer.candidate_dataset.len() > max_candidates {
        let remove_count = trainer.candidate_dataset.len() - max_candidates;
        trainer.candidate_dataset.drain(0..remove_count);
    }

    if max_paths > 0 && trainer.paths.len() > max_paths {
        trainer.paths.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        trainer.paths.truncate(max_paths);
    }
}

fn compact_trainer_for_checkpoint(trainer: &RawAITrainer) -> RawAITrainer {
    let mut compact = trainer.clone_for_checkpoint();

    for grid in &mut compact.grids {
        for page in &mut grid.pages {
            for case in &mut page.cases {
                case.signal = 0.0;
                case.gradient = 0.0;
            }
        }
    }

    for correlation in &mut compact.correlations {
        correlation.coefficient_gradient = 0.0;
        correlation.last_activity = 0.0;
    }

    apply_memory_limits(&mut compact);
    compact
}

fn estimate_trainer_memory_bytes(trainer: &RawAITrainer) -> usize {
    let mut bytes = std::mem::size_of::<RawAITrainer>();
    let pages = trainer.grids.iter().map(|grid| grid.pages.len()).sum::<usize>();
    let cases = trainer
        .grids
        .iter()
        .flat_map(|grid| &grid.pages)
        .map(|page| page.cases.len())
        .sum::<usize>();
    bytes += trainer.grids.capacity() * std::mem::size_of::<Grid>();
    bytes += pages * std::mem::size_of::<GridPage>();
    bytes += cases * std::mem::size_of::<Case>();
    bytes += trainer.correlations.capacity() * std::mem::size_of::<Correlation>();
    bytes += trainer.paths.capacity() * std::mem::size_of::<PreferredPath>();
    bytes += trainer.pack_state.queue.len() * std::mem::size_of::<PackJob>();
    bytes += trainer.candidate_dataset.len() * std::mem::size_of::<TeacherDataCandidate>();
    bytes
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let value = bytes as f64;
    if value >= GB {
        format!("{:.2} GiB", value / GB)
    } else if value >= MB {
        format!("{:.2} MiB", value / MB)
    } else if value >= KB {
        format!("{:.2} KiB", value / KB)
    } else {
        format!("{bytes} B")
    }
}

fn run_memory_diagnostics() -> Result<(), Box<dyn Error>> {
    let mut trainer = new_configured_trainer();
    configure_demo_graph(&mut trainer);
    trainer.pack_state.max_items_per_theme = env_usize("BRICKS_AI_ITEMS_PER_THEME").unwrap_or(1);
    trainer.training_pack.max_depth = env_usize("BRICKS_AI_MAX_DEPTH").unwrap_or(0);
    trainer.start_pack_training();
    apply_memory_limits(&mut trainer);

    let pages = trainer.grids.iter().map(|grid| grid.pages.len()).sum::<usize>();
    let cases = trainer
        .grids
        .iter()
        .flat_map(|grid| &grid.pages)
        .map(|page| page.cases.len())
        .sum::<usize>();

    println!("Bricks AI memory diagnostics");
    println!("memory_mode={:?}", MemoryMode::from_env());
    println!("grids={}", trainer.grids.len());
    println!("pages={pages}");
    println!("cases={cases}");
    println!("correlations={}", trainer.correlations.len());
    println!("paths={}", trainer.paths.len());
    println!("queue_jobs={}", trainer.pack_state.queue.len());
    println!("candidate_cache={}", trainer.candidate_dataset.len());
    println!("estimated_trainer_memory={}", format_bytes(estimate_trainer_memory_bytes(&trainer)));
    println!("settings: BRICKS_AI_MEMORY_MODE, BRICKS_AI_GRID_COUNT, BRICKS_AI_CASES_PER_GRID, BRICKS_AI_MAX_QUEUE_JOBS, BRICKS_AI_MAX_CANDIDATE_CACHE, BRICKS_AI_MAX_PATH_CACHE");
    Ok(())
}

fn run_export_demo() -> Result<(), Box<dyn Error>> {
    let mut trainer = RawAITrainer::new(2, 32);
    let input = trainer.official_node(0, 3);
    let output = trainer.official_node(1, 7);
    trainer.add_correlation(input, output, 0.4);

    for _ in 0..20 {
        trainer.train_step(&[(input, 1.0)], &[(output, 1.0)], 1);
    }

    trainer.engrave_validated_weights(0.01, 0.01);
    let path = "engraved_model.json";
    trainer.save_engraved_model(path)?;
    println!("Saved {path}");
    Ok(())
}

fn configure_demo_graph(trainer: &mut RawAITrainer) -> (NodeId, NodeId) {
    trainer.case_learning_rate = 0.005;
    trainer.correlation_learning_rate = 0.01;
    trainer.affinity_learning_rate = 0.0005;

    let (input, output) = demo_input_output_nodes(trainer);
    let mid_1 = trainer.official_node(1, safe_case_index(trainer, 1, 45));
    let mid_2 = trainer.official_node(2, safe_case_index(trainer, 2, 300));

    trainer.add_correlation(input, mid_1, 0.20);
    trainer.add_correlation(mid_1, mid_2, 0.20);
    trainer.add_correlation(mid_2, output, 0.20);
    trainer.discover_paths_from(input, 4, 8);

    (input, output)
}

fn demo_input_output_nodes(trainer: &RawAITrainer) -> (NodeId, NodeId) {
    let input = trainer.official_node(0, safe_case_index(trainer, 0, 12));
    let output_grid = trainer.grids.len().saturating_sub(1).max(1);
    let output = trainer.official_node(output_grid, safe_case_index(trainer, output_grid, 900));
    (input, output)
}

fn safe_case_index(trainer: &RawAITrainer, grid: usize, preferred: usize) -> usize {
    let page = trainer.grids[grid].official_page;
    let len = trainer.grids[grid].pages[page].cases.len();
    preferred.min(len.saturating_sub(1))
}

#[derive(Debug, Clone, Copy)]
enum ApiProtocol {
    OllamaGenerate,
    OpenAiChat,
    AnthropicMessages,
    GeminiGenerateContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderMode {
    /// Try Ollama first on every request, then cloud providers if Ollama fails.
    LocalFirst,
    /// Use only Ollama. Cloud providers are ignored even if API keys exist.
    OllamaOnly,
    /// Use only cloud providers. Ollama is ignored even if enabled.
    CloudOnly,
    /// Rotate across every available provider.
    RoundRobin,
}

impl ProviderMode {
    fn from_env() -> Self {
        let value = env::var("BRICKS_AI_PROVIDER_MODE")
            .unwrap_or_else(|_| "local-first".to_string())
            .to_ascii_lowercase();

        match value.as_str() {
            "ollama-only" | "local-only" => Self::OllamaOnly,
            "cloud-only" | "api-only" => Self::CloudOnly,
            "round-robin" | "balanced" => Self::RoundRobin,
            _ => Self::LocalFirst,
        }
    }
}

#[derive(Debug, Clone)]
struct TeacherProviderClient {
    provider: AiProvider,
    name: &'static str,
    key_env: &'static str,
    model: String,
    api_key: String,
    endpoint: String,
    protocol: ApiProtocol,
    http: Client,
    min_interval: Duration,
    requests_per_minute: usize,
    rate_limit_cooldown: Duration,
    failure_cooldown: Duration,
    last_request_at: Option<Instant>,
    request_history: VecDeque<Instant>,
    cooldown_until: Option<Instant>,
    consecutive_failures: usize,
}

impl TeacherProviderClient {
    fn display_name(&self) -> String {
        format!("{} [{:?}]", self.name, self.provider)
    }

    fn cooldown_remaining(&self) -> Option<Duration> {
        let cooldown_until = self.cooldown_until?;
        cooldown_until.checked_duration_since(Instant::now())
    }

    fn wait_for_rate_slot(&mut self) {
        if let Some(wait) = self.cooldown_remaining() {
            thread::sleep(wait);
        }

        if let Some(last_request_at) = self.last_request_at {
            if let Some(wait) = self.min_interval.checked_sub(last_request_at.elapsed()) {
                thread::sleep(wait);
            }
        }

        let now = Instant::now();
        while let Some(oldest) = self.request_history.front().copied() {
            if now.duration_since(oldest) >= Duration::from_secs(60) {
                self.request_history.pop_front();
            } else {
                break;
            }
        }

        if self.requests_per_minute > 0 && self.request_history.len() >= self.requests_per_minute {
            if let Some(oldest) = self.request_history.front().copied() {
                if let Some(wait) = Duration::from_secs(60).checked_sub(oldest.elapsed()) {
                    thread::sleep(wait);
                }
            }
            self.request_history.clear();
        }
    }

    fn register_request(&mut self) {
        let now = Instant::now();
        self.last_request_at = Some(now);
        self.request_history.push_back(now);
    }

    fn register_success(&mut self) {
        self.consecutive_failures = 0;
        self.cooldown_until = None;
    }

    fn register_failure(&mut self, error: &str) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let base = if is_rate_limit_error(error) {
            self.rate_limit_cooldown
        } else {
            self.failure_cooldown
        };
        let multiplier = self.consecutive_failures.min(4) as u64;
        let cooldown_seconds = base.as_secs().saturating_mul(multiplier.max(1));
        let cooldown = Duration::from_secs(cooldown_seconds.max(1));
        self.cooldown_until = Some(Instant::now() + cooldown);
    }

    fn call_text(&mut self, prompt: &str) -> Result<String, String> {
        self.wait_for_rate_slot();
        self.register_request();

        match self.protocol {
            ApiProtocol::OllamaGenerate => self.call_ollama_generate(prompt),
            ApiProtocol::OpenAiChat => self.call_openai_compatible_chat(prompt),
            ApiProtocol::AnthropicMessages => self.call_anthropic_messages(prompt),
            ApiProtocol::GeminiGenerateContent => self.call_gemini_generate_content(prompt),
        }
    }

    fn call_ollama_generate(&self, prompt: &str) -> Result<String, String> {
        let endpoint = format!(
            "{}/api/generate",
            self.endpoint.trim_end_matches('/')
        );

        let payload = build_ollama_payload(&self.model, prompt);

        let response = self
            .http
            .post(&endpoint)
            .json(&payload)
            .send()
            .map_err(|e| format!("{} local connection error at {}: {:?}", self.display_name(), endpoint, e))?;

        let status = response.status();
        let body = response.text().map_err(|e| format!("{} response read error: {:?}", self.display_name(), e))?;

        if !status.is_success() {
            return Err(format!("{} local API error {}: {}", self.display_name(), status, body));
        }

        let value: Value = serde_json::from_str(&body)
            .map_err(|e| format!("{} invalid JSON response: {}. Raw: {}", self.display_name(), e, first_chars(&body, 1200)))?;

        value
            .get("response")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("{} response did not contain `response`: {}", self.display_name(), first_chars(&body, 1200)))
    }

    fn call_openai_compatible_chat(&self, prompt: &str) -> Result<String, String> {
        let response = self
            .http
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a strict data generation and validation engine for Bricks AI. Return only the requested JSON when JSON is requested."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }))
            .send()
            .map_err(|e| format!("{} network error: {}", self.display_name(), e))?;

        let status = response.status();
        let body = response.text().map_err(|e| format!("{} response read error: {}", self.display_name(), e))?;

        if !status.is_success() {
            return Err(format!("{} API error {}: {}", self.display_name(), status, body));
        }

        let value: Value = serde_json::from_str(&body).map_err(|e| format!("{} invalid JSON response: {}", self.display_name(), e))?;
        extract_chat_text(&value).ok_or_else(|| format!("{} response did not contain message content: {}", self.display_name(), body))
    }

    fn call_anthropic_messages(&self, prompt: &str) -> Result<String, String> {
        let response = self
            .http
            .post(&self.endpoint)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&json!({
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }))
            .send()
            .map_err(|e| format!("{} network error: {}", self.display_name(), e))?;

        let status = response.status();
        let body = response.text().map_err(|e| format!("{} response read error: {}", self.display_name(), e))?;

        if !status.is_success() {
            return Err(format!("{} API error {}: {}", self.display_name(), status, body));
        }

        let value: Value = serde_json::from_str(&body).map_err(|e| format!("{} invalid JSON response: {}", self.display_name(), e))?;
        extract_anthropic_text(&value).ok_or_else(|| format!("{} response did not contain text content: {}", self.display_name(), body))
    }

    fn call_gemini_generate_content(&self, prompt: &str) -> Result<String, String> {
        let endpoint = format!("{}{}:generateContent?key={}", self.endpoint, self.model, self.api_key);
        let response = self
            .http
            .post(endpoint)
            .json(&json!({
                "contents": [
                    {"parts": [{"text": prompt}]}
                ],
                "generationConfig": {"temperature": 0.2}
            }))
            .send()
            .map_err(|e| format!("{} network error: {}", self.display_name(), e))?;

        let status = response.status();
        let body = response.text().map_err(|e| format!("{} response read error: {}", self.display_name(), e))?;

        if !status.is_success() {
            return Err(format!("{} API error {}: {}", self.display_name(), status, body));
        }

        let value: Value = serde_json::from_str(&body).map_err(|e| format!("{} invalid JSON response: {}", self.display_name(), e))?;
        extract_gemini_text(&value).ok_or_else(|| format!("{} response did not contain text content: {}", self.display_name(), body))
    }
}

#[derive(Debug)]
struct TeacherProviderPool {
    providers: Vec<TeacherProviderClient>,
    next_index: usize,
    mode: ProviderMode,
    debug_provider_fallbacks: bool,
}

impl TeacherProviderPool {
    fn from_env() -> Self {
        let http = build_cloud_http_client();
        let ollama_http = build_ollama_http_client();

        let mut providers = Vec::new();

        push_ollama_provider(&mut providers, ollama_http);

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::OpenAI,
            "OpenAI",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "gpt-5.5",
            "https://api.openai.com/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::Anthropic,
            "Anthropic",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL",
            "claude-sonnet-4-5",
            "https://api.anthropic.com/v1/messages",
            ApiProtocol::AnthropicMessages,
        );

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::Gemini,
            "Gemini",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "gemini-2.5-pro",
            "https://generativelanguage.googleapis.com/v1beta/models/",
            ApiProtocol::GeminiGenerateContent,
        );

        if env::var("GEMINI_API_KEY").ok().filter(|v| !v.trim().is_empty()).is_none() {
            push_provider(
                &mut providers,
                http.clone(),
                AiProvider::Gemini,
                "Gemini",
                "GOOGLE_API_KEY",
                "GEMINI_MODEL",
                "gemini-2.5-pro",
                "https://generativelanguage.googleapis.com/v1beta/models/",
                ApiProtocol::GeminiGenerateContent,
            );
        }

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::Mistral,
            "Mistral",
            "MISTRAL_API_KEY",
            "MISTRAL_MODEL",
            "mistral-large-latest",
            "https://api.mistral.ai/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::Xai,
            "xAI",
            "XAI_API_KEY",
            "XAI_MODEL",
            "grok-4",
            "https://api.x.ai/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::DeepSeek,
            "DeepSeek",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_MODEL",
            "deepseek-chat",
            "https://api.deepseek.com/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        push_provider(
            &mut providers,
            http.clone(),
            AiProvider::Groq,
            "Groq",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
            "https://api.groq.com/openai/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        push_provider(
            &mut providers,
            http,
            AiProvider::Together,
            "Together",
            "TOGETHER_API_KEY",
            "TOGETHER_MODEL",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "https://api.together.xyz/v1/chat/completions",
            ApiProtocol::OpenAiChat,
        );

        let mode = ProviderMode::from_env();

        match mode {
            ProviderMode::OllamaOnly => {
                providers.retain(|provider| provider.provider == AiProvider::Ollama);
            }
            ProviderMode::CloudOnly => {
                providers.retain(|provider| provider.provider != AiProvider::Ollama);
            }
            ProviderMode::LocalFirst | ProviderMode::RoundRobin => {}
        }

        // Keep Ollama at the front in local-first mode. This ensures local inference
        // is attempted before paid APIs on every request.
        if mode == ProviderMode::LocalFirst {
            providers.sort_by_key(|provider| if provider.provider == AiProvider::Ollama { 0 } else { 1 });
        }

        let debug_provider_fallbacks = env_bool("BRICKS_AI_DEBUG_PROVIDERS").unwrap_or(false);

        Self {
            providers,
            next_index: 0,
            mode,
            debug_provider_fallbacks,
        }
    }

    fn generate_parsed<T, F>(&mut self, prompt: &str, mut parser: F) -> Result<(String, T), String>
    where
        F: FnMut(&str) -> Result<T, String>,
    {
        if self.providers.is_empty() {
            return Err(format!(
                "no providers available for mode {:?}. Enable Ollama with OLLAMA_ENABLED=true or add a cloud API key.",
                self.mode
            ));
        }

        let mut errors = Vec::new();
        let len = self.providers.len();

        let order: Vec<usize> = match self.mode {
            ProviderMode::LocalFirst | ProviderMode::OllamaOnly | ProviderMode::CloudOnly => (0..len).collect(),
            ProviderMode::RoundRobin => (0..len).map(|attempt| (self.next_index + attempt) % len).collect(),
        };

        for index in order {
            if let Some(wait) = self.providers[index].cooldown_remaining() {
                let message = format!(
                    "{} cooling down for {}s",
                    self.providers[index].display_name(),
                    wait.as_secs().max(1)
                );

                if self.debug_provider_fallbacks {
                    println!("provider_skip={message}");
                }

                errors.push(message);
                continue;
            }

            let provider_name = self.providers[index].display_name();

            if self.debug_provider_fallbacks {
                println!("provider_try={provider_name}");
            }

            let response = self.providers[index].call_text(prompt);

            match response {
                Ok(text) => match parser(&text) {
                    Ok(parsed) => {
                        self.providers[index].register_success();

                        self.next_index = match self.mode {
                            ProviderMode::RoundRobin => (index + 1) % len,
                            ProviderMode::LocalFirst | ProviderMode::OllamaOnly | ProviderMode::CloudOnly => 0,
                        };

                        return Ok((provider_name, parsed));
                    }
                    Err(parse_error) => {
                        let retry_prompt = build_json_repair_prompt(prompt, &text, &parse_error);

                        if self.debug_provider_fallbacks {
                            println!("provider_retry_json={provider_name} reason={parse_error}");
                        }

                        match self.providers[index].call_text(&retry_prompt) {
                            Ok(retry_text) => match parser(&retry_text) {
                                Ok(parsed) => {
                                    self.providers[index].register_success();
                                    self.next_index = match self.mode {
                                        ProviderMode::RoundRobin => (index + 1) % len,
                                        ProviderMode::LocalFirst | ProviderMode::OllamaOnly | ProviderMode::CloudOnly => 0,
                                    };
                                    return Ok((provider_name, parsed));
                                }
                                Err(retry_parse_error) => {
                                    let preview = retry_text.chars().take(300).collect::<String>().replace('\n', " ");
                                    let message = format!(
                                        "{provider_name} parse error after retry: {retry_parse_error}. response_preview={preview}"
                                    );
                                    self.providers[index].register_failure(&message);

                                    if self.debug_provider_fallbacks {
                                        println!("provider_fallback={message}");
                                    }

                                    errors.push(message);
                                }
                            },
                            Err(retry_call_error) => {
                                let preview = text.chars().take(300).collect::<String>().replace('\n', " ");
                                let message = format!(
                                    "{provider_name} parse error: {parse_error}. retry_call_error={retry_call_error}. response_preview={preview}"
                                );
                                self.providers[index].register_failure(&message);

                                if self.debug_provider_fallbacks {
                                    println!("provider_fallback={message}");
                                }

                                errors.push(message);
                            }
                        }
                    }
                },
                Err(call_error) => {
                    self.providers[index].register_failure(&call_error);

                    if self.debug_provider_fallbacks {
                        println!("provider_fallback={call_error}");
                    }

                    errors.push(call_error);
                }
            }
        }

        Err(errors.join("\n"))
    }

}


fn build_json_repair_prompt(original_prompt: &str, invalid_response: &str, parse_error: &str) -> String {
    format!(
        "Your previous answer was not parseable JSON. Error: {parse_error}\n\nReturn ONLY raw JSON in the exact shape requested by the original task. No markdown. No code fences. No explanations. No text before or after the JSON.\n\nOriginal task:\n{original}\n\nInvalid response to repair:\n{invalid}\n",
        parse_error = parse_error,
        original = first_chars(original_prompt, 4000),
        invalid = first_chars(invalid_response, 4000)
    )
}

fn push_ollama_provider(providers: &mut Vec<TeacherProviderClient>, http: Client) {
    if !env_bool("OLLAMA_ENABLED").unwrap_or(false) {
        return;
    }

    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".to_string());
    let endpoint = env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());

    let prefix = provider_env_prefix(AiProvider::Ollama);
    let min_interval = Duration::from_secs(provider_env_usize(prefix, "MIN_SECONDS_BETWEEN_REQUESTS", default_min_seconds(AiProvider::Ollama)) as u64);
    let requests_per_minute = provider_env_usize(prefix, "REQUESTS_PER_MINUTE", default_requests_per_minute(AiProvider::Ollama)).max(1);
    let rate_limit_cooldown = Duration::from_secs(provider_env_usize(prefix, "RATE_LIMIT_COOLDOWN_SECONDS", 10) as u64);
    let failure_cooldown = Duration::from_secs(provider_env_usize(prefix, "FAILURE_COOLDOWN_SECONDS", 5) as u64);

    providers.push(TeacherProviderClient {
        provider: AiProvider::Ollama,
        name: "Ollama",
        key_env: "OLLAMA_ENABLED",
        model,
        api_key: String::new(),
        endpoint,
        protocol: ApiProtocol::OllamaGenerate,
        http,
        min_interval,
        requests_per_minute,
        rate_limit_cooldown,
        failure_cooldown,
        last_request_at: None,
        request_history: VecDeque::new(),
        cooldown_until: None,
        consecutive_failures: 0,
    });
}

fn push_provider(
    providers: &mut Vec<TeacherProviderClient>,
    http: Client,
    provider: AiProvider,
    name: &'static str,
    key_env: &'static str,
    model_env: &'static str,
    default_model: &'static str,
    endpoint: &'static str,
    protocol: ApiProtocol,
) {
    let Some(api_key) = env::var(key_env).ok().filter(|value| !value.trim().is_empty()) else {
        return;
    };

    let model = env::var(model_env).unwrap_or_else(|_| default_model.to_string());

    let prefix = provider_env_prefix(provider);
    let min_interval = Duration::from_secs(provider_env_usize(prefix, "MIN_SECONDS_BETWEEN_REQUESTS", default_min_seconds(provider)) as u64);
    let requests_per_minute = provider_env_usize(prefix, "REQUESTS_PER_MINUTE", default_requests_per_minute(provider)).max(1);
    let rate_limit_cooldown = Duration::from_secs(provider_env_usize(prefix, "RATE_LIMIT_COOLDOWN_SECONDS", 90) as u64);
    let failure_cooldown = Duration::from_secs(provider_env_usize(prefix, "FAILURE_COOLDOWN_SECONDS", 15) as u64);

    providers.push(TeacherProviderClient {
        provider,
        name,
        key_env,
        model,
        api_key,
        endpoint: endpoint.to_string(),
        protocol,
        http,
        min_interval,
        requests_per_minute,
        rate_limit_cooldown,
        failure_cooldown,
        last_request_at: None,
        request_history: VecDeque::new(),
        cooldown_until: None,
        consecutive_failures: 0,
    });
}

fn provider_env_prefix(provider: AiProvider) -> &'static str {
    match provider {
        AiProvider::Ollama => "OLLAMA",
        AiProvider::OpenAI => "OPENAI",
        AiProvider::Anthropic => "ANTHROPIC",
        AiProvider::Gemini => "GEMINI",
        AiProvider::Mistral => "MISTRAL",
        AiProvider::Xai => "XAI",
        AiProvider::DeepSeek => "DEEPSEEK",
        AiProvider::Groq => "GROQ",
        AiProvider::Together => "TOGETHER",
    }
}

fn default_min_seconds(provider: AiProvider) -> usize {
    match provider {
        AiProvider::Ollama => 1,
        AiProvider::OpenAI => 6,
        AiProvider::Anthropic => 10,
        AiProvider::Gemini => 6,
        AiProvider::Mistral => 8,
        AiProvider::Xai => 10,
        AiProvider::DeepSeek => 8,
        AiProvider::Groq => 6,
        AiProvider::Together => 6,
    }
}

fn default_requests_per_minute(provider: AiProvider) -> usize {
    match provider {
        AiProvider::Ollama => 60,
        AiProvider::OpenAI => 10,
        AiProvider::Anthropic => 6,
        AiProvider::Gemini => 10,
        AiProvider::Mistral => 7,
        AiProvider::Xai => 6,
        AiProvider::DeepSeek => 8,
        AiProvider::Groq => 10,
        AiProvider::Together => 10,
    }
}

fn provider_env_usize(prefix: &str, suffix: &str, default_value: usize) -> usize {
    let provider_key = format!("{prefix}_{suffix}");
    env_usize(&provider_key)
        .or_else(|| env_usize(&format!("BRICKS_AI_{suffix}")))
        .unwrap_or(default_value)
}

fn is_rate_limit_error(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("429")
        || lower.contains("too many requests")
        || lower.contains("rate limit")
        || lower.contains("rate_limited")
        || lower.contains("insufficient_quota")
}

fn validate_items_with_pool(
    pool: &mut TeacherProviderPool,
    job: &PackJob,
    items: &[ApiTeacherItem],
) -> Result<(String, HashMap<usize, f32>), String> {
    let compact_items: Vec<Value> = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            json!({
                "index": index,
                "question": item.question,
                "answer": item.answer,
                "confidence": item.confidence,
                "tags": item.tags,
                "verification_notes": item.verification_notes
            })
        })
        .collect();

    let theme = job.theme_path.join(" > ");
    let prompt = format!(
        "You are a strict validator for Bricks AI training data.\n\nTheme: {theme}\n\nScore every item for relevance, correctness, educational value, and safety. Return only valid JSON in this shape:\n[{{\"index\":0,\"validator_score\":0.0,\"accepted\":false,\"notes\":\"...\"}}]\n\nReject uncertain, speculative, unsafe, personalized high-stakes, or off-theme items.\n\nItems:\n{}",
        serde_json::to_string_pretty(&compact_items).map_err(|e| e.to_string())?
    );

    pool.generate_parsed(&prompt, |text| {
        let verdicts: Vec<ApiValidationVerdict> = parse_json_array(text).map_err(|e| e.to_string())?;
        let scores = verdicts
            .into_iter()
            .map(|verdict| {
                let noted = !verdict.notes.trim().is_empty();
                let score = if verdict.accepted { verdict.validator_score } else { 0.0 };
                let adjusted_score = if noted { score } else { score * 0.98 };
                (verdict.index, adjusted_score.clamp(0.0, 1.0))
            })
            .collect();
        Ok(scores)
    })
}

#[derive(Debug, Deserialize)]
struct ApiSubtheme {
    name: String,
    #[serde(default)]
    safety: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ApiTeacherItem {
    question: String,
    answer: String,
    confidence: f32,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    verification_notes: String,
}

impl ApiTeacherItem {
    fn into_core(self) -> PackTeacherItem {
        PackTeacherItem {
            question: self.question,
            answer: self.answer,
            confidence: self.confidence.clamp(0.0, 1.0),
            tags: self.tags,
            verification_notes: self.verification_notes,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiValidationVerdict {
    index: usize,
    validator_score: f32,
    #[serde(default)]
    accepted: bool,
    #[serde(default)]
    notes: String,
}

fn parse_subthemes(text: &str) -> Result<Vec<GeneratedSubTheme>, Box<dyn Error>> {
    let subthemes: Vec<ApiSubtheme> = parse_json_array(text)?;
    Ok(subthemes
        .into_iter()
        .filter(|item| !item.name.trim().is_empty())
        .map(|item| GeneratedSubTheme {
            name: item.name.trim().to_string(),
            safety: parse_safety(item.safety.as_deref().unwrap_or("normal")),
        })
        .collect())
}

fn parse_teacher_items(text: &str) -> Result<Vec<ApiTeacherItem>, Box<dyn Error>> {
    let items: Vec<ApiTeacherItem> = parse_json_array(text)?;
    Ok(items
        .into_iter()
        .filter(|item| !item.question.trim().is_empty() && !item.answer.trim().is_empty())
        .map(|mut item| {
            item.confidence = item.confidence.clamp(0.0, 1.0);
            item
        })
        .collect())
}

fn parse_safety(value: &str) -> SafetyLevel {
    let normalized = value.to_ascii_lowercase();
    if normalized.contains("high") || normalized.contains("sensitive") {
        SafetyLevel::HighStakes
    } else {
        SafetyLevel::Normal
    }
}


fn parse_json_object<T>(text: &str) -> Result<T, Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    let trimmed = text.trim();
    if let Ok(item) = serde_json::from_str::<T>(trimmed) {
        return Ok(item);
    }

    let candidates = extract_json_object_candidates(trimmed);
    let mut errors = Vec::new();
    for candidate in candidates {
        match serde_json::from_str::<T>(&candidate) {
            Ok(item) => return Ok(item),
            Err(error) => errors.push(format!(
                "candidate failed: {error}; candidate_preview={}",
                first_chars(&candidate.replace('\n', " "), 220)
            )),
        }
    }

    Err(boxed_error(format!(
        "provider response did not contain a valid JSON object. response_preview={}. parser_errors={}",
        first_chars(&trimmed.replace('\n', " "), 700),
        if errors.is_empty() { "none".to_string() } else { errors.join(" | ") }
    )))
}

fn extract_json_object_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();
    for (start, ch) in text.char_indices() {
        if ch != '{' {
            continue;
        }
        if let Some(end) = find_matching_json_object_end(text, start) {
            let candidate = text[start..=end].trim().to_string();
            if !candidates.iter().any(|existing| existing == &candidate) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.len()));
    candidates
}

fn find_matching_json_object_end(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (offset, ch) in text[start..].char_indices() {
        let index = start + offset;
        if in_string {
            if escape {
                escape = false;
                continue;
            }
            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => depth = depth.saturating_add(1),
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_json_array<T>(text: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: for<'de> Deserialize<'de>,
{
    let trimmed = text.trim();
    if let Ok(items) = serde_json::from_str::<Vec<T>>(trimmed) {
        return Ok(items);
    }

    let candidates = extract_json_array_candidates(trimmed);
    let mut errors = Vec::new();

    for candidate in candidates {
        match serde_json::from_str::<Vec<T>>(&candidate) {
            Ok(items) => return Ok(items),
            Err(error) => errors.push(format!(
                "candidate failed: {error}; candidate_preview={}",
                first_chars(&candidate.replace('\n', " "), 220)
            )),
        }
    }

    Err(boxed_error(format!(
        "provider response did not contain a valid JSON array. response_preview={}. parser_errors={}",
        first_chars(&trimmed.replace('\n', " "), 700),
        if errors.is_empty() { "none".to_string() } else { errors.join(" | ") }
    )))
}

fn extract_json_array_candidates(text: &str) -> Vec<String> {
    let mut candidates = Vec::new();

    for (start, ch) in text.char_indices() {
        if ch != '[' {
            continue;
        }

        if let Some(end) = find_matching_json_array_end(text, start) {
            let candidate = text[start..=end].trim().to_string();
            if !candidates.iter().any(|existing| existing == &candidate) {
                candidates.push(candidate);
            }
        }
    }

    if let Some(repaired) = salvage_complete_objects_from_partial_array(text) {
        if !candidates.iter().any(|existing| existing == &repaired) {
            candidates.push(repaired);
        }
    }

    // Prefer larger arrays first because nested arrays like `tags: [...]` are not
    // the top-level data shape we want. This makes Ollama markdown-wrapped JSON
    // parse correctly even when it adds text before or after the array.
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.len()));
    candidates
}

fn salvage_complete_objects_from_partial_array(text: &str) -> Option<String> {
    let array_start = text.find('[')?;
    let mut objects = Vec::new();

    let mut in_string = false;
    let mut escape = false;
    let mut object_depth = 0usize;
    let mut object_start: Option<usize> = None;

    for (offset, ch) in text[array_start..].char_indices() {
        let index = array_start + offset;

        if in_string {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' => {
                if object_depth == 0 {
                    object_start = Some(index);
                }
                object_depth += 1;
            }
            '}' => {
                if object_depth == 0 {
                    continue;
                }
                object_depth -= 1;
                if object_depth == 0 {
                    if let Some(start) = object_start.take() {
                        objects.push(text[start..=index].trim().to_string());
                    }
                }
            }
            _ => {}
        }
    }

    if objects.is_empty() {
        None
    } else {
        Some(format!("[{}]", objects.join(",")))
    }
}

fn find_matching_json_array_end(text: &str, start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape = false;

    for (offset, ch) in text[start..].char_indices() {
        let index = start + offset;

        if in_string {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' => depth = depth.saturating_add(1),
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_chat_text(value: &Value) -> Option<String> {
    if let Some(text) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
    {
        return Some(text.to_string());
    }

    recursive_find_text(value)
}

fn extract_anthropic_text(value: &Value) -> Option<String> {
    let content = value.get("content")?.as_array()?;
    let mut parts = Vec::new();
    for item in content {
        if let Some(text) = item.get("text").and_then(Value::as_str) {
            parts.push(text.to_string());
        }
    }
    if parts.is_empty() { recursive_find_text(value) } else { Some(parts.join("\n")) }
}

fn extract_gemini_text(value: &Value) -> Option<String> {
    let candidates = value.get("candidates")?.as_array()?;
    let first = candidates.first()?;
    let parts = first.get("content")?.get("parts")?.as_array()?;
    let mut out = Vec::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            out.push(text.to_string());
        }
    }
    if out.is_empty() { recursive_find_text(value) } else { Some(out.join("\n")) }
}

fn recursive_find_text(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => {
            if let Some(text) = map.get("text").and_then(Value::as_str) {
                return Some(text.to_string());
            }
            for child in map.values() {
                if let Some(text) = recursive_find_text(child) {
                    return Some(text);
                }
            }
            None
        }
        Value::Array(items) => {
            for item in items {
                if let Some(text) = recursive_find_text(item) {
                    return Some(text);
                }
            }
            None
        }
        _ => None,
    }
}

#[derive(Serialize, Deserialize)]
struct TrainingCheckpoint {
    completed_steps: usize,
    trainer: RawAITrainer,
}

fn save_checkpoint(path: &str, completed_steps: usize, trainer: &RawAITrainer) -> Result<(), Box<dyn Error>> {
    let checkpoint = TrainingCheckpoint {
        completed_steps,
        trainer: compact_trainer_for_checkpoint(trainer),
    };
    let bytes = bincode::serialize(&checkpoint)?;
    fs::write(path, bytes)?;
    Ok(())
}

fn load_checkpoint(path: &str) -> Result<TrainingCheckpoint, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(bincode::deserialize(&bytes)?)
}

trait TrainerCheckpointClone {
    fn clone_for_checkpoint(&self) -> RawAITrainer;
}

impl TrainerCheckpointClone for RawAITrainer {
    fn clone_for_checkpoint(&self) -> RawAITrainer {
        RawAITrainer {
            grids: self.grids.clone(),
            correlations: self.correlations.clone(),
            adjacency: self.adjacency.clone(),
            paths: self.paths.clone(),
            groups: self.groups.clone(),
            critical_zones: self.critical_zones.clone(),
            teacher_settings: self.teacher_settings.clone(),
            candidate_dataset: self.candidate_dataset.clone(),
            training_pack: self.training_pack.clone(),
            pack_state: PackTrainingState {
                status: self.pack_state.status,
                queue: self.pack_state.queue.clone(),
                current_job: self.pack_state.current_job.clone(),
                accepted_items: self.pack_state.accepted_items,
                rejected_items: self.pack_state.rejected_items,
                trained_items: self.pack_state.trained_items,
                max_items_per_theme: self.pack_state.max_items_per_theme,
            },
            dimensions: self.dimensions.clone(),
            dimension_paths: self.dimension_paths.clone(),
            convergence_cube: self.convergence_cube.clone(),
            pre_final_destruction: self.pre_final_destruction.clone(),
            case_learning_rate: self.case_learning_rate,
            correlation_learning_rate: self.correlation_learning_rate,
            affinity_learning_rate: self.affinity_learning_rate,
            prune_threshold: self.prune_threshold,
            max_coefficient: self.max_coefficient,
        }
    }
}

fn flag_usize(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .and_then(|window| window[1].parse::<usize>().ok())
}


fn build_cloud_http_client() -> Client {
    Client::builder()
        .timeout(Duration::from_secs(env_usize("BRICKS_AI_TIMEOUT_SECONDS").unwrap_or(120) as u64))
        .build()
        .expect("failed to build cloud HTTP client")
}

fn build_ollama_http_client() -> Client {
    Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(env_usize("OLLAMA_TIMEOUT_SECONDS").unwrap_or(300) as u64))
        .build()
        .expect("failed to build Ollama HTTP client")
}

fn build_ollama_payload(model: &str, prompt: &str) -> Value {
    let mut payload = json!({
        "model": model,
        "prompt": prompt,
        "stream": false,
        "options": {
            "temperature": 0.0,
            "num_predict": env_usize("OLLAMA_NUM_PREDICT").unwrap_or(512)
        }
    });

    if env_bool("OLLAMA_JSON_FORMAT").unwrap_or(false) {
        payload["format"] = json!("json");
    }

    payload
}

fn first_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn flag_present(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse::<usize>().ok())
}

fn env_f32(name: &str) -> Option<f32> {
    env::var(name).ok().and_then(|value| value.parse::<f32>().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().and_then(|value| match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

fn first_lines(text: &str, max_lines: usize) -> String {
    text.lines().take(max_lines).collect::<Vec<_>>().join("\n")
}

fn boxed_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::other(message.into()))
}
