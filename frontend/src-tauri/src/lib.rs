#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .setup(|app| {
      #[cfg(desktop)]
      {
        use tauri_plugin_shell::ShellExt;
        
        // Kill any existing backend process on port 8000
        let _ = std::process::Command::new("sh")
            .arg("-c")
            .arg("lsof -ti :8000 | xargs kill -9 || true")
            .output();

        let sidecar = app.shell().sidecar("backend").unwrap();
        let (mut rx, mut _child) = sidecar.spawn().expect("Failed to spawn backend");

        tauri::async_runtime::spawn(async move {
          while let Some(event) = rx.recv().await {
            use tauri_plugin_shell::process::CommandEvent;
            match event {
               CommandEvent::Stdout(line) => {
                  let text = String::from_utf8(line).unwrap();
                  log::info!("Backend stdout: {}", text);
               },
               CommandEvent::Stderr(line) => {
                  let text = String::from_utf8(line).unwrap();
                  log::error!("Backend stderr: {}", text);
               },
               _ => {}
            }
          }
        });
      }

      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
