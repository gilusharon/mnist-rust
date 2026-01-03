use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use candle_datasets::vision::mnist;
use clap::{Parser, ValueEnum};
use eframe::egui;
use image::{GrayImage, ImageBuffer, Luma};

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

fn test(model: &Model, test_images: &Tensor, test_labels: &Tensor) -> Result<()> {
    println!("Evaluating on test set...");
    
    let logits = model.forward(test_images)?;
    let predictions = logits.argmax(1).map_err(anyhow::Error::from)?;    
    let test_labels_1d = if test_labels.rank() > 1 {
        test_labels.flatten_all().map_err(anyhow::Error::from)?
    } else {
        test_labels.clone()
    };
    let test_labels_u32 = test_labels_1d.to_dtype(candle_core::DType::U32).map_err(anyhow::Error::from)?;
    let correct = predictions
        .eq(&test_labels_u32).map_err(anyhow::Error::from)?
        .to_vec1::<u8>().map_err(anyhow::Error::from)?;
    
    let accuracy = correct.iter().map(|&x| x as f64).sum::<f64>() / correct.len() as f64;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

struct DrawingApp {
    canvas: Vec<Vec<bool>>,
    is_drawing: bool,
    brush_size: f32,
}

impl DrawingApp {
    fn new() -> Self {
        Self {
            canvas: vec![vec![false; 280]; 280], // 280x280 canvas, will be downscaled to 28x28
            is_drawing: false,
            brush_size: 10.0,
        }
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

    fn save_as_jpeg(&self, filename: &str) -> Result<()> {
        let mut img: GrayImage = ImageBuffer::new(28, 28);
        
        for y in 0..28 {
            for x in 0..28 {
                // Sample from 10x10 region in the canvas
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
                
                let avg = if count > 0 { (sum / count) as u8 } else { 0 };
                img.put_pixel(x as u32, y as u32, Luma([avg]));
            }
        }
        
        // Save as JPEG
        img.save(filename)?;
        println!("Saved drawing to {}", filename);
        Ok(())
    }
}

impl eframe::App for DrawingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Draw a Digit (28x28)");
            
            ui.horizontal(|ui| {
                ui.label("Brush Size:");
                ui.add(egui::Slider::new(&mut self.brush_size, 5.0..=30.0));
            });
            
            ui.horizontal(|ui| {
                if ui.button("Clear").clicked() {
                    self.clear();
                }
                
                if ui.button("Save as JPEG").clicked() {
                    if let Err(e) = self.save_as_jpeg("drawn_digit.jpg") {
                        eprintln!("Error saving: {}", e);
                    }
                }
            });
            
            ui.separator();
            
            // Drawing canvas
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(280.0, 280.0),
                egui::Sense::click_and_drag(),
            );
            
            let canvas_rect = response.rect;
            painter.rect_filled(canvas_rect, 0.0, egui::Color32::WHITE);
            
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
            
            // Handle mouse input
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
            
            // Draw the canvas pixels
            for y in 0..280 {
                for x in 0..280 {
                    if self.canvas[y][x] {
                        let pixel_rect = egui::Rect::from_min_size(
                            canvas_rect.min + egui::vec2(x as f32, y as f32),
                            egui::Vec2::new(1.0, 1.0),
                        );
                        painter.rect_filled(pixel_rect, 0.0, egui::Color32::BLACK);
                    }
                }
            }
        });
        
        ctx.request_repaint();
    }
}

fn draw_digit() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 500.0])
            .with_title("Draw a Digit"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Draw a Digit",
        options,
        Box::new(|_cc| Box::new(DrawingApp::new())),
    ).map_err(|e| anyhow::anyhow!("Failed to run drawing app: {}", e))?;
    
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    let dataset = mnist::load().map_err(anyhow::Error::from)?;
    let train_images = dataset.train_images.to_device(&device)?.to_dtype(candle_core::DType::F32)?;
    let test_images = dataset.test_images.to_device(&device)?.to_dtype(candle_core::DType::F32)?;
    let train_labels = dataset.train_labels.to_device(&device)?.to_dtype(candle_core::DType::U32)?;
    let test_labels = dataset.test_labels.to_device(&device)?.to_dtype(candle_core::DType::U32)?;
    let mut varmap = VarMap::new();
    let model = Model::new(VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device))?;
    train(&model, &train_images, &train_labels, args.epochs, args.batch_size, args.learning_rate, &mut varmap)?;
    test(&model, &test_images, &test_labels)?;
    let tensors = varmap.data().lock().unwrap().iter()
    .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
    .collect::<std::collections::HashMap<String, Tensor>>();
    candle_core::safetensors::save(&tensors, "model.safetensors")?;
    Ok(())
}

