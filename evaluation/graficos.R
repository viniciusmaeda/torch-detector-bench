library(ggplot2)
library(dplyr)
library(tidyr)

options(scipen = 999)

# ============================================================
# Configurações gerais
# ============================================================

base_dir <- "/home/vinicius/Documentos/results_bioparque_20260514"
output_dir <- file.path(base_dir, "figures_boxplots")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat("Working directory:", base_dir, "\n")
cat("Output directory:", output_dir, "\n\n")

# ============================================================
# Arquivos e nomes dos modelos
# ============================================================

model_files <- tibble::tribble(
  ~file,                                          ~model,
  "yolov8m_history.csv",                         "YOLOv8m",
  "yolo11m_history.csv",                         "YOLO11m",
  "yolo26m_history.csv",                         "YOLO26m",
  "fasterrcnn_resnet50_fpn_history.csv",         "Faster R-CNN FPN",
  "fasterrcnn_resnet50_fpn_v2_history.csv",      "Faster R-CNN FPN v2",
  "fasterrcnn_mobilenet_v3_large_fpn_history.csv","Faster R-CNN MobileNetV3",
  "detr_history.csv",                            "DETR",
  "rtdetr_l_history.csv",                        "RT-DETR"
)

model_order <- model_files$model

# ============================================================
# Verificação dos arquivos
# ============================================================

model_files <- model_files %>%
  mutate(path = file.path(base_dir, file))

missing_files <- model_files %>%
  filter(!file.exists(path))

if (nrow(missing_files) > 0) {
  stop(
    paste0(
      "The following files were not found:\n",
      paste(missing_files$file, collapse = "\n")
    )
  )
}

cat("All CSV files were found successfully.\n\n")

# ============================================================
# Função para carregar dados de cada modelo
# ============================================================

load_model_data <- function(path, model_name) {
  
  df <- read.csv(path, check.names = TRUE)
  
  # Caso 1: arquivos no padrão Ultralytics YOLO / RT-DETR
  if ("metrics.precision.B." %in% names(df)) {
    
    precision <- df$metrics.precision.B.
    recall <- df$metrics.recall.B.
    map50 <- df$metrics.mAP50.B.
    map95 <- df$metrics.mAP50.95.B.
    f1 <- 2 * (precision * recall) / (precision + recall + 1e-9)
    
    # Caso 2: arquivos no padrão Faster R-CNN / DETR
  } else if (all(c("precision", "recall", "f1", "map50", "map50_95") %in% names(df))) {
    
    precision <- df$precision
    recall <- df$recall
    f1 <- df$f1
    map50 <- df$map50
    map95 <- df$map50_95
    
  } else {
    
    stop(
      paste0(
        "Unexpected column format in file: ", basename(path), "\n",
        "Available columns are:\n",
        paste(names(df), collapse = ", ")
      )
    )
  }
  
  data.frame(
    model = model_name,
    Precision = precision,
    Recall = recall,
    `F1-score` = f1,
    `mAP@0.5` = map50,
    `mAP@0.5:0.95` = map95,
    check.names = FALSE
  )
}

# ============================================================
# Carregar e organizar dados
# ============================================================

dados <- bind_rows(
  lapply(seq_len(nrow(model_files)), function(i) {
    load_model_data(
      path = model_files$path[i],
      model_name = model_files$model[i]
    )
  })
)

dados$model <- factor(dados$model, levels = model_order)

dados_long <- dados %>%
  pivot_longer(
    cols = c("Precision", "Recall", "F1-score", "mAP@0.5", "mAP@0.5:0.95"),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Metric = factor(
      Metric,
      levels = c("Precision", "Recall", "F1-score", "mAP@0.5", "mAP@0.5:0.95")
    )
  )

# ============================================================
# Função para gerar boxplot
# ============================================================

save_boxplot <- function(metric_name, data_long, output_dir) {
  
  df_plot <- data_long %>%
    filter(Metric == metric_name)
  
  safe_name <- metric_name %>%
    gsub("@", "", .) %>%
    gsub(":", "_", .) %>%
    gsub("-", "_", .)
  
  g <- ggplot(df_plot, aes(x = model, y = Value, fill = model)) +
    geom_boxplot(outlier.size = 1.5, linewidth = 0.4) +
    scale_fill_brewer(palette = "Purples") +
    labs(
      title = paste("Boxplot for", metric_name),
      x = "Models",
      y = metric_name
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5),
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid.minor = element_blank()
    )
  
  png_file <- file.path(output_dir, paste0("boxplot_", safe_name, ".png"))
  pdf_file <- file.path(output_dir, paste0("boxplot_", safe_name, ".pdf"))
  
  ggsave(png_file, g, width = 7, height = 5, dpi = 300)
  ggsave(pdf_file, g, width = 7, height = 5)
  
  print(g)
  
  cat("Saved:", png_file, "\n")
  cat("Saved:", pdf_file, "\n\n")
}

# ============================================================
# Gerar gráficos individuais
# ============================================================

metrics_to_plot <- c("Precision", "Recall", "F1-score", "mAP@0.5", "mAP@0.5:0.95")

for (metric in metrics_to_plot) {
  save_boxplot(metric, dados_long, output_dir)
}