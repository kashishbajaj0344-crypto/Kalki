use std::collections::HashMap;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager, State};
use tokio::process::Command as AsyncCommand;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub kalki_running: bool,
    pub server_running: bool,
    pub agents_active: Vec<String>,
    pub consciousness_level: f64,
    pub system_health: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub name: String,
    pub status: String,
    pub capabilities: Vec<String>,
    pub last_activity: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    pub awareness_level: f64,
    pub emotional_resonance: f64,
    pub self_reflection_depth: u32,
    pub intention_coherence: f64,
    pub neural_patterns: HashMap<String, f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EngineeringTask {
    pub id: String,
    pub name: String,
    pub status: String,
    pub progress: f64,
    pub result: Option<String>,
}

pub struct AppState {
    pub kalki_process: Arc<Mutex<Option<std::process::Child>>>,
    pub server_process: Arc<Mutex<Option<std::process::Child>>>,
    pub system_status: Arc<Mutex<SystemStatus>>,
    pub agents: Arc<Mutex<HashMap<String, AgentInfo>>>,
    pub consciousness_metrics: Arc<Mutex<ConsciousnessMetrics>>,
    pub engineering_tasks: Arc<Mutex<HashMap<String, EngineeringTask>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            kalki_process: Arc::new(Mutex::new(None)),
            server_process: Arc::new(Mutex::new(None)),
            system_status: Arc::new(Mutex::new(SystemStatus {
                kalki_running: false,
                server_running: false,
                agents_active: vec![],
                consciousness_level: 0.0,
                system_health: "unknown".to_string(),
            })),
            agents: Arc::new(Mutex::new(HashMap::new())),
            consciousness_metrics: Arc::new(Mutex::new(ConsciousnessMetrics {
                awareness_level: 0.0,
                emotional_resonance: 0.0,
                self_reflection_depth: 0,
                intention_coherence: 0.0,
                neural_patterns: HashMap::new(),
            })),
            engineering_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

// System Management Commands
#[tauri::command]
async fn start_kalki_system(
    app: AppHandle,
    state: State<'_, AppState>
) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    // Start Kalki main system
    let mut cmd = Command::new("python3");
    cmd.arg("kalki.py")
        .current_dir(&kalki_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(child) => {
            *state.kalki_process.lock().unwrap() = Some(child);
            let mut status = state.system_status.lock().unwrap();
            status.kalki_running = true;
            Ok("Kalki system started successfully".to_string())
        }
        Err(e) => Err(format!("Failed to start Kalki system: {}", e)),
    }
}

#[tauri::command]
async fn stop_kalki_system(state: State<'_, AppState>) -> Result<String, String> {
    if let Some(mut child) = state.kalki_process.lock().unwrap().take() {
        match child.kill() {
            Ok(_) => {
                let mut status = state.system_status.lock().unwrap();
                status.kalki_running = false;
                Ok("Kalki system stopped successfully".to_string())
            }
            Err(e) => Err(format!("Failed to stop Kalki system: {}", e)),
        }
    } else {
        Ok("Kalki system was not running".to_string())
    }
}

#[tauri::command]
async fn start_kalki_server(
    app: AppHandle,
    state: State<'_, AppState>
) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    // Start Flask server
    let mut cmd = Command::new("python3");
    cmd.arg("kalki_server.py")
        .current_dir(&kalki_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    match cmd.spawn() {
        Ok(child) => {
            *state.server_process.lock().unwrap() = Some(child);
            let mut status = state.system_status.lock().unwrap();
            status.server_running = true;
            Ok("Kalki server started successfully".to_string())
        }
        Err(e) => Err(format!("Failed to start Kalki server: {}", e)),
    }
}

#[tauri::command]
async fn get_system_status(state: State<'_, AppState>) -> Result<SystemStatus, String> {
    let status = state.system_status.lock().unwrap().clone();
    Ok(status)
}

#[tauri::command]
async fn get_agents_info(state: State<'_, AppState>) -> Result<Vec<AgentInfo>, String> {
    let agents = state.agents.lock().unwrap();
    let agents_list: Vec<AgentInfo> = agents.values().cloned().collect();
    Ok(agents_list)
}

#[tauri::command]
async fn get_consciousness_metrics(state: State<'_, AppState>) -> Result<ConsciousnessMetrics, String> {
    let metrics = state.consciousness_metrics.lock().unwrap().clone();
    Ok(metrics)
}

// Engineering Commands
#[tauri::command]
async fn run_engineering_task(
    app: AppHandle,
    task_type: String,
    parameters: HashMap<String, String>,
    state: State<'_, AppState>
) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;
    let task_id = format!("task_{}", chrono::Utc::now().timestamp());

    // Create task entry
    let task = EngineeringTask {
        id: task_id.clone(),
        name: task_type.clone(),
        status: "running".to_string(),
        progress: 0.0,
        result: None,
    };

    state.engineering_tasks.lock().unwrap().insert(task_id.clone(), task);

    // Run task in background
    let state_clone = state.inner().clone();
    let kalki_path_clone = kalki_path.clone();
    let task_type_clone = task_type.clone();
    let parameters_clone = parameters.clone();

    let task_id_clone = task_id.clone();

    let state_clone = state.engineering_tasks.clone();

    tokio::spawn(async move {
        let result = execute_engineering_task(
            &kalki_path_clone,
            &task_type_clone,
            &parameters_clone
        ).await;

        let mut tasks = state_clone.lock().unwrap();
        if let Some(task) = tasks.get_mut(&task_id_clone) {
            task.status = if result.is_ok() { "completed".to_string() } else { "failed".to_string() };
            task.progress = 100.0;
            task.result = Some(result.unwrap_or_else(|e| e));
        }
    });

    Ok(task_id)
}

#[tauri::command]
async fn get_engineering_tasks(state: State<'_, AppState>) -> Result<Vec<EngineeringTask>, String> {
    let tasks = state.engineering_tasks.lock().unwrap();
    let tasks_list: Vec<EngineeringTask> = tasks.values().cloned().collect();
    Ok(tasks_list)
}

// Knowledge Base Commands
#[tauri::command]
async fn search_knowledge_base(
    app: AppHandle,
    query: String,
    limit: Option<usize>
) -> Result<Vec<HashMap<String, String>>, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    let output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.rag_query import search_knowledge; import json; print(json.dumps(search_knowledge('{}', {})))", query, limit.unwrap_or(10)))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| format!("Failed to execute search: {}", e))?;

    if output.status.success() {
        let results: Vec<HashMap<String, String>> = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("Failed to parse results: {}", e))?;
        Ok(results)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Search failed: {}", stderr))
    }
}

// Demo Commands
#[tauri::command]
async fn run_consciousness_demo(app: AppHandle) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    let output = AsyncCommand::new("python3")
        .arg("demo_consciousness_bootstrap.py")
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| format!("Failed to run consciousness demo: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Demo failed: {}", stderr))
    }
}

#[tauri::command]
async fn run_iron_man_design(app: AppHandle) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    let output = AsyncCommand::new("python3")
        .arg("design_iron_man_suit.py")
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| format!("Failed to run Iron Man design: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Design failed: {}", stderr))
    }
}

// Utility Functions
fn get_kalki_project_path(app: &AppHandle) -> Result<PathBuf, String> {
    let app_dir = app.path().app_data_dir()
        .map_err(|e| format!("Failed to get app data dir: {}", e))?;

    // Go up from kalki-desktop to Kalki root
    let kalki_path = app_dir
        .parent()
        .ok_or("Invalid app directory structure")?
        .to_path_buf();

    Ok(kalki_path)
}

async fn execute_engineering_task(
    kalki_path: &PathBuf,
    task_type: &str,
    parameters: &HashMap<String, String>
) -> Result<String, String> {
    match task_type {
        "robotics_simulation" => {
            let output = AsyncCommand::new("python3")
                .arg("-c")
                .arg("from modules.agents.core.robotics_simulation import RoboticsSimulationAgent; import asyncio; asyncio.run(RoboticsSimulationAgent().execute({}))")
                .current_dir(kalki_path)
                .output()
                .await
                .map_err(|e| e.to_string())?;
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        "cad_design" => {
            let output = AsyncCommand::new("python3")
                .arg("-c")
                .arg("from modules.agents.core.cad_integration import CADIntegrationAgent; import asyncio; asyncio.run(CADIntegrationAgent().execute({}))")
                .current_dir(kalki_path)
                .output()
                .await
                .map_err(|e| e.to_string())?;
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        }
        _ => Err(format!("Unknown task type: {}", task_type)),
    }
}

// AI Orchestrator Command
#[tauri::command]
async fn orchestrate_task(
    app: AppHandle,
    request: String,
    platform: String,
    state: State<'_, AppState>
) -> Result<String, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    // Step 1: Analyze request with LLM
    let analysis_output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.llm import LLMEngine; llm = LLMEngine(); result = llm.analyze_request('{}'); print(result)", request))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    let analysis = String::from_utf8_lossy(&analysis_output.stdout);

    // Step 2: Research requirements using web search and vector DB
    let research_output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.vectordb import VectorDBManager; from modules.web_search import WebSearchAPI; vdb = VectorDBManager(); ws = WebSearchAPI(); research = vdb.search_similar('{}'); web_results = ws.search('{} requirements {}'); print({{ 'vector_db': research, 'web_search': web_results }})", request, platform, request))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    // Step 3: Delegate to appropriate agents based on platform and requirements
    let delegation_result = match platform.as_str() {
        "unity" => {
            // For Unity games, delegate to multiple agents
            let robotics_result = AsyncCommand::new("python3")
                .arg("-c")
                .arg(format!("from modules.agents.core.robotics_simulation import RoboticsSimulationAgent; agent = RoboticsSimulationAgent(); result = agent.execute({{ 'task': 'game_physics', 'platform': 'unity', 'request': '{}' }}); print(result)", request))
                .current_dir(&kalki_path)
                .output()
                .await
                .map_err(|e| e.to_string())?;

            let cad_result = AsyncCommand::new("python3")
                .arg("-c")
                .arg(format!("from modules.agents.core.cad_integration import CADIntegrationAgent; agent = CADIntegrationAgent(); result = agent.execute({{ 'task': '3d_assets', 'platform': 'unity', 'request': '{}' }}); print(result)", request))
                .current_dir(&kalki_path)
                .output()
                .await
                .map_err(|e| e.to_string())?;

            format!("Unity Game Development:\nPhysics: {}\nAssets: {}",
                   String::from_utf8_lossy(&robotics_result.stdout),
                   String::from_utf8_lossy(&cad_result.stdout))
        }
        "web" => {
            // For web apps, use different agents
            let llm_result = AsyncCommand::new("python3")
                .arg("-c")
                .arg(format!("from modules.llm import LLMEngine; llm = LLMEngine(); result = llm.generate_code('{}', 'web'); print(result)", request))
                .current_dir(&kalki_path)
                .output()
                .await
                .map_err(|e| e.to_string())?;

            String::from_utf8_lossy(&llm_result.stdout).to_string()
        }
        _ => format!("Platform '{}' orchestration completed. Analysis: {}", platform, analysis)
    };

    // Step 4: Validate results with LLM
    let validation_output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.llm import LLMEngine; llm = LLMEngine(); validation = llm.validate_result('{}', '{}'); print(validation)", delegation_result, request))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    let validation = String::from_utf8_lossy(&validation_output.stdout);

    Ok(format!("🎯 Orchestration Complete!\n\n📋 Request: {}\n🎮 Platform: {}\n\n🔍 Analysis: {}\n\n🧠 Agent Results: {}\n\n✅ Validation: {}\n\n🚀 Ready for deployment!",
              request, platform, analysis, delegation_result, validation))
}

// Analyze orchestrator request and determine if clarification is needed
#[tauri::command]
async fn analyze_orchestrator_request(
    app: AppHandle,
    request: String,
    platform: String
) -> Result<serde_json::Value, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    // Use LLM to analyze the request and determine what information is missing
    let analysis_output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.llm import LLMEngine; llm = LLMEngine(); analysis = llm.analyze_request_clarification('{}', '{}'); print(analysis)", request, platform))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    let analysis_str = String::from_utf8_lossy(&analysis_output.stdout);
    let analysis: serde_json::Value = serde_json::from_str(&analysis_str)
        .map_err(|e| format!("Failed to parse analysis: {}", e))?;

    Ok(analysis)
}

// Continue orchestration with user answers
#[tauri::command]
async fn continue_orchestration_with_answers(
    app: AppHandle,
    request: String,
    platform: String,
    answers: Vec<serde_json::Value>
) -> Result<serde_json::Value, String> {
    let kalki_path = get_kalki_project_path(&app)?;

    // Process the answers and continue with orchestration
    let answers_json = serde_json::to_string(&answers)
        .map_err(|e| format!("Failed to serialize answers: {}", e))?;

    let result_output = AsyncCommand::new("python3")
        .arg("-c")
        .arg(format!("from modules.llm import LLMEngine; llm = LLMEngine(); result = llm.process_clarifications('{}', '{}', '{}'); print(result)", request, platform, answers_json))
        .current_dir(&kalki_path)
        .output()
        .await
        .map_err(|e| e.to_string())?;

    let result_str = String::from_utf8_lossy(&result_output.stdout);
    let result: serde_json::Value = serde_json::from_str(&result_str)
        .map_err(|e| format!("Failed to parse result: {}", e))?;

    Ok(result)
}

// Legacy greet command for compatibility
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! Welcome to Kalki AI Framework!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            greet,
            start_kalki_system,
            stop_kalki_system,
            start_kalki_server,
            get_system_status,
            get_agents_info,
            get_consciousness_metrics,
            run_engineering_task,
            get_engineering_tasks,
            search_knowledge_base,
            run_consciousness_demo,
            run_iron_man_design,
            orchestrate_task,
            analyze_orchestrator_request,
            continue_orchestration_with_answers
        ])
        .run(tauri::generate_context!())
        .expect("error while running Kalki Desktop application");
}
