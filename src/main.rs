use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use candle_datasets::vision::mnist;
use clap::{Parser, ValueEnum};
use eframe::egui;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(value_enum, default_value_t = Mode::Train)]
    mode: Mode,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    #[arg(long, default_value_t = 0.01)]
    learning_rate: f64,
}

#[derive(Clone, ValueEnum)]
enum Mode {
    Train,
    Test,
    Draw,
}

struct Model {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(784, 128, vs.pp("linear1"))?;
        let linear2 = candle_nn::linear(128, 64, vs.pp("linear2"))?;
        let linear3 = candle_nn::linear(64, 10, vs.pp("linear3"))?;
        Ok(Self {
            linear1,
            linear2,
            linear3,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.flatten_from(1).map_err(anyhow::Error::from)?;
        let xs = self.linear1.forward(&xs).map_err(anyhow::Error::from)?;
        let xs = xs.relu().map_err(anyhow::Error::from)?;
        let xs = self.linear2.forward(&xs).map_err(anyhow::Error::from)?;
        let xs = xs.relu().map_err(anyhow::Error::from)?;
        let xs = self.linear3.forward(&xs).map_err(anyhow::Error::from)?;
        Ok(xs)
    }
}

fn train(
    model: &Model,
    train_images: &Tensor,
    train_labels: &Tensor,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    varmap: &mut VarMap,
) -> Result<()> {
    let train_size = train_images.dim(0).map_err(anyhow::Error::from)?;
    let n_batches = train_size / batch_size;
    let train_vars = varmap.all_vars();
    let mut sgd = candle_nn::SGD::new(train_vars, learning_rate)?;
    
    for epoch in 0..epochs {
        let mut sum_loss = 0f64;
        let mut min_loss = f32::MAX;
        let mut max_loss = f32::MIN;

        for batch_idx in 0..n_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(train_size);
            let batch_size_actual = end_idx - start_idx;

            let batch_images = train_images.narrow(0, start_idx, batch_size_actual).map_err(anyhow::Error::from)?;
            let batch_labels = train_labels.narrow(0, start_idx, batch_size_actual).map_err(anyhow::Error::from)?;
            let logits= model.forward(&batch_images)?;
            let loss = loss::cross_entropy(&logits, &batch_labels)
                .map_err(anyhow::Error::from)?;

            sgd.backward_step(&loss).unwrap();

            let loss_val = loss.to_vec0::<f32>().map_err(anyhow::Error::from)?;
            sum_loss += loss_val as f64;
            min_loss = min_loss.min(loss_val);
            max_loss = max_loss.max(loss_val);
        }

        let avg_loss = sum_loss / n_batches as f64;
        println!("Epoch {}/{} - Average Loss: {:.4} (min: {:.4}, max: {:.4})", 
                 epoch + 1, epochs, avg_loss, min_loss, max_loss);
    }

    Ok(())
}

struct DrawingApp {
    canvas: Vec<Vec<bool>>,
    is_drawing: bool,
    brush_size: f32,
    device: Device,
    dataset: Option<(Tensor, Tensor, Tensor, Tensor)>,
    varmap: Arc<Mutex<VarMap>>,
    model: Option<Arc<Model>>,
    training_status: String,
    prediction: Option<u32>,
}

impl DrawingApp {
    fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        Ok(Self {
            canvas: vec![vec![false; 280]; 280],
            is_drawing: false,
            brush_size: 10.0,
            device,
            dataset: None,
            varmap: Arc::new(Mutex::new(VarMap::new())),
            model: None,
            training_status: String::new(),
            prediction: None,
        })
    }

    fn clear(&mut self) {
        self.canvas = vec![vec![false; 280]; 280];
    }

    fn draw_pixel(&mut self, x: usize, y: usize) {
        let radius = self.brush_size as usize / 2;
        for dy in 0..=radius * 2 {
            for dx in 0..=radius * 2 {
                let px = x as i32 + dx as i32 - radius as i32;
                let py = y as i32 + dy as i32 - radius as i32;
                if px >= 0 && px < 280 && py >= 0 && py < 280 {
                    let dist = ((dx as f32 - radius as f32).powi(2) + (dy as f32 - radius as f32).powi(2)).sqrt();
                    if dist <= radius as f32 {
                        self.canvas[py as usize][px as usize] = true;
                    }
                }
            }
        }
    }

    fn canvas_to_tensor(&self) -> Result<Tensor> {
        // Convert canvas to 28x28 grayscale values (0.0 to 1.0)
        // Invert colors: drawn pixels (black in UI) become white (1.0), background becomes black (0.0)
        let mut pixels = Vec::new();
        for y in 0..28 {
            for x in 0..28 {
                let mut sum = 0u32;
                let mut count = 0u32;
                
                for sy in 0..10 {
                    for sx in 0..10 {
                        let canvas_x = x * 10 + sx;
                        let canvas_y = y * 10 + sy;
                        if canvas_x < 280 && canvas_y < 280 {
                            if self.canvas[canvas_y][canvas_x] {
                                sum += 255;
                            }
                            count += 1;
                        }
                    }
                }
                
                // Invert: drawn pixels become 1.0 (white), background becomes 0.0 (black)
                let avg = if count > 0 { (sum / count) as f32 / 255.0 } else { 0.0 };
                pixels.push(avg);
            }
        }
        
        // Create tensor: shape [1, 28, 28] for a single image
        Tensor::from_slice(&pixels, &[1, 28, 28], &self.device)
            .map_err(anyhow::Error::from)
    }

    fn identify_digit(&mut self) -> Result<()> {
        if self.model.is_none() {
            self.training_status = "Error: Please load or train a model first".to_string();
            return Ok(());
        }

        let image_tensor = self.canvas_to_tensor()?;
        let model = self.model.as_ref().unwrap();
        
        let logits = model.forward(&image_tensor)?;
        let prediction_tensor = logits.argmax(1).map_err(anyhow::Error::from)?;
        
        // argmax(1) returns shape [1], so we need to get the first element
        let prediction = prediction_tensor
            .to_vec1::<u32>()
            .map_err(anyhow::Error::from)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty prediction tensor"))?;
        
        self.prediction = Some(prediction);
        self.training_status = format!("Identified as: {}", prediction);
        Ok(())
    }

    fn load_model(&mut self) -> Result<()> {
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &self.device);
        let model = Model::new(vs)?;       
        let tensors = candle_core::safetensors::load("model.safetensors", &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
        
        varmap.set(tensors.iter().map(|(k, v)| (k, v)))
            .map_err(|e| anyhow::anyhow!("Failed to set model parameters: {}", e))?;
        
        self.varmap = Arc::new(Mutex::new(varmap));
        self.model = Some(Arc::new(model));
        self.training_status = "Model loaded successfully".to_string();
        Ok(())
    }

    fn train_model(&mut self, epochs: usize, batch_size: usize, learning_rate: f64) {
        // Load dataset if not already loaded
        if self.dataset.is_none() {
            self.training_status = "Loading dataset...".to_string();
            let dataset = match mnist::load() {
                Ok(d) => d,
                Err(e) => {
                    self.training_status = format!("Error loading dataset: {}", e);
                    return;
                }
            };
            
            let train_images = match dataset.train_images.to_device(&self.device) {
                Ok(img) => img,
                Err(e) => {
                    self.training_status = format!("Error moving train images to device: {}", e);
                    return;
                }
            };
            let test_images = match dataset.test_images.to_device(&self.device) {
                Ok(img) => img,
                Err(e) => {
                    self.training_status = format!("Error moving test images to device: {}", e);
                    return;
                }
            };
            let train_labels = match dataset.train_labels.to_device(&self.device) {
                Ok(lbl) => lbl,
                Err(e) => {
                    self.training_status = format!("Error moving train labels to device: {}", e);
                    return;
                }
            };
            let test_labels = match dataset.test_labels.to_device(&self.device) {
                Ok(lbl) => lbl,
                Err(e) => {
                    self.training_status = format!("Error moving test labels to device: {}", e);
                    return;
                }
            };
            
            let train_images = match train_images.to_dtype(candle_core::DType::F32) {
                Ok(img) => img,
                Err(e) => {
                    self.training_status = format!("Error converting train images: {}", e);
                    return;
                }
            };
            let test_images = match test_images.to_dtype(candle_core::DType::F32) {
                Ok(img) => img,
                Err(e) => {
                    self.training_status = format!("Error converting test images: {}", e);
                    return;
                }
            };
            let train_labels = match train_labels.to_dtype(candle_core::DType::U32) {
                Ok(lbl) => lbl,
                Err(e) => {
                    self.training_status = format!("Error converting train labels: {}", e);
                    return;
                }
            };
            let test_labels = match test_labels.to_dtype(candle_core::DType::U32) {
                Ok(lbl) => lbl,
                Err(e) => {
                    self.training_status = format!("Error converting test labels: {}", e);
                    return;
                }
            };
            
            self.dataset = Some((train_images, train_labels, test_images, test_labels));
            self.training_status = "Dataset loaded, starting training...".to_string();
        }

        let (train_images, train_labels, _, _) = self.dataset.as_ref().unwrap();
        let mut varmap_guard = self.varmap.lock().unwrap();
        
        if self.model.is_none() {
            let vs = VarBuilder::from_varmap(&*varmap_guard, candle_core::DType::F32, &self.device);
            match Model::new(vs) {
                Ok(model) => {
                    self.model = Some(Arc::new(model));
                }
                Err(e) => {
                    self.training_status = format!("Error creating model: {}", e);
                    return;
                }
            }
        }

        let model = self.model.as_ref().unwrap();
        
        match train(model, train_images, train_labels, epochs, batch_size, learning_rate, &mut *varmap_guard) {
            Ok(()) => {
                let tensors = varmap_guard.data().lock().unwrap().iter()
                    .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
                    .collect::<std::collections::HashMap<String, Tensor>>();
                
                if let Err(e) = candle_core::safetensors::save(&tensors, "model.safetensors") {
                    self.training_status = format!("Error saving model: {}", e);
                } else {
                    self.training_status = format!("Training completed and model saved");
                }
            }
            Err(e) => {
                self.training_status = format!("Training error: {}", e);
            }
        }
    }
}

impl eframe::App for DrawingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("MNIST Classifier - Draw & Train");
            
            ui.horizontal(|ui| {
                if ui.button("Load Model").clicked() {
                    if let Err(e) = self.load_model() {
                        self.training_status = format!("Error loading model: {}", e);
                    }
                }
                
                if ui.button("Train").clicked() {
                    self.train_model(10, 64, 0.01);
                }
            });
            
            if !self.training_status.is_empty() {
                ui.label(&self.training_status);
            }
            
            ui.separator();
            
            ui.horizontal(|ui| {
                ui.label("Brush Size:");
                ui.add(egui::Slider::new(&mut self.brush_size, 5.0..=30.0));
            });
            
            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    self.clear();
                    self.prediction = None;
                }
                
                if ui.button("Identify").clicked() {
                    if let Err(e) = self.identify_digit() {
                        self.training_status = format!("Error identifying: {}", e);
                    }
                }
            });
            
            if let Some(pred) = self.prediction {
                ui.heading(format!("Prediction: {}", pred));
            }
            
            ui.separator();
            
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(280.0, 280.0),
                egui::Sense::click_and_drag(),
            );
            
            let canvas_rect = response.rect;
            painter.rect_filled(canvas_rect, 0.0, egui::Color32::BLACK);
            
            painter.line_segment(
                [canvas_rect.min, egui::pos2(canvas_rect.min.x, canvas_rect.max.y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
            );
            painter.line_segment(
                [canvas_rect.min, egui::pos2(canvas_rect.max.x, canvas_rect.min.y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
            );
            painter.line_segment(
                [canvas_rect.max, egui::pos2(canvas_rect.min.x, canvas_rect.max.y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
            );
            painter.line_segment(
                [canvas_rect.max, egui::pos2(canvas_rect.max.x, canvas_rect.min.y)],
                egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
            );
            
            if response.dragged() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    let local_pos = pointer_pos - canvas_rect.min;
                    let x = local_pos.x as usize;
                    let y = local_pos.y as usize;
                    
                    if x < 280 && y < 280 {
                        self.is_drawing = true;
                        self.draw_pixel(x, y);
                    }
                }
            } else if response.clicked() {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    let local_pos = pointer_pos - canvas_rect.min;
                    let x = local_pos.x as usize;
                    let y = local_pos.y as usize;
                    
                    if x < 280 && y < 280 {
                        self.is_drawing = true;
                        self.draw_pixel(x, y);
                    }
                }
            } else {
                self.is_drawing = false;
            }
            
            for y in 0..280 {
                for x in 0..280 {
                    if self.canvas[y][x] {
                        let pixel_rect = egui::Rect::from_min_size(
                            canvas_rect.min + egui::vec2(x as f32, y as f32),
                            egui::Vec2::new(1.0, 1.0),
                        );
                        painter.rect_filled(pixel_rect, 0.0, egui::Color32::WHITE);
                    }
                }
            }
        });
        
        ctx.request_repaint();
    }
}

fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 600.0])
            .with_title("MNIST Classifier"),
        ..Default::default()
    };
    
    eframe::run_native(
        "MNIST Classifier",
        options,
        Box::new(|_cc| {
            Box::new(DrawingApp::new().expect("Failed to create app"))
        }),
    ).map_err(|e| anyhow::anyhow!("Failed to run app: {:?}", e))?;
    
    Ok(())
}

