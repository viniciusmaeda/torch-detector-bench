# Boxplots descritivos de métricas FINAIS por dobra; épocas não são replicações.
library(ggplot2)
library(dplyr)
library(tidyr)

args <- commandArgs(trailingOnly = TRUE)
metrics_file <- if (length(args)) args[[1]] else "results/fish_cv5_seed42/reports/fold_metrics.csv"
if (!file.exists(metrics_file)) stop(paste("Arquivo não encontrado:", metrics_file))
output_dir <- file.path(dirname(metrics_file), "figures_R")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

data <- read.csv(metrics_file, check.names = FALSE)
metrics <- c("mAP50_95", "mAP50", "Precision", "Recall", "F1", "MAE", "RMSE", "FPS")
if (!all(c("Model", "Fold", metrics) %in% names(data))) stop("CSV incompatível: use reports/fold_metrics.csv")
if (anyDuplicated(data[c("Model", "Fold")])) stop("Há resultados repetidos para modelo/dobra")

long <- data %>%
  mutate(Fold = factor(Fold), Model = factor(Model, levels = unique(Model))) %>%
  pivot_longer(cols = all_of(metrics), names_to = "Metric", values_to = "Value")

for (metric in metrics) {
  selected <- filter(long, Metric == metric)
  graph <- ggplot(selected, aes(x = Model, y = Value)) +
    geom_boxplot(width = 0.5, outlier.shape = NA) +
    geom_line(aes(group = Fold, color = Fold), alpha = 0.5) +
    geom_point(aes(color = Fold), size = 2) +
    labs(title = metric, subtitle = "Avaliação final por dobra; linhas indicam pareamento",
         x = "Modelo", y = metric) +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1), panel.grid.minor = element_blank())
  for (extension in c("png", "pdf", "svg")) {
    device <- switch(extension, png = "png", pdf = "pdf", svg = grDevices::svg)
    ggsave(file.path(output_dir, paste0("fold_", metric, ".", extension)), graph,
           device = device, width = 10, height = 6, dpi = 300)
  }
}
cat("Gráficos salvos em", output_dir, "\n")
