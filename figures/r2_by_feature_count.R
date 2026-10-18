library(ggplot2)
library(tidyverse)
library(dplyr)

# --- Paths --- (relative to repo root, run: Rscript figures/r2_by_feature_count.R)
svr_p <- "results/svm_brute_force/svm_feature_subset_results.csv"
rf_p  <- "results/feature_subset_results.csv"
dir.create("results/plots", showWarnings = FALSE, recursive = TRUE)

# --- Load & preprocess ---
parse_features <- function(x) {
  x |> str_remove_all("\\[|\\]|'") |> str_split(",\\s*")
}

rf_df  <- read.csv(rf_p)  |> mutate(model = "RF",  features = parse_features(features))
svr_df <- read.csv(svr_p) |> mutate(model = "SVR", features = parse_features(features))

results <- bind_rows(
  rf_df  |> select(model, features, n_features, RMSE, MAE, R2),
  svr_df |> select(model, features, n_features, RMSE, MAE, R2)
)

# --- Boxplot: R² by feature count (top 70% models) ---
results_top70 <- results %>%
  group_by(model) %>%
  filter(RMSE <= quantile(RMSE, 0.70)) %>%
  ungroup()

ggplot(
  results_top70,
  aes(x = factor(n_features), y = R2, fill = model)
) +
  geom_boxplot(
    outlier.alpha = 0.2,
    position = position_dodge(width = 0.8)
  ) +
  labs(
    title = "R² Distribution by Number of Features \n(Top 70% Models)",
    x = "Number of features",
    y = "R²",
    fill = "Model"
  ) +
  theme_grey(base_size = 22) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 24, face = "bold"),
    axis.title = element_text(size = 20, face = "bold"),
    axis.text  = element_text(size = 18, face="bold"),
    legend.position  = c(0.9, 0.85),
    legend.text      = element_text(size = 16, face = "bold"),
    legend.title     = element_text(size = 18, face = "bold"),
    legend.background = element_rect(
      fill = "grey90",
      alpha=0.1,
      color = "black",
      linewidth = 0.5
    )
  )

ggsave(
  "results/plots/R2_by_feature_count.png",
  width = 10,
  height = 6,
  dpi = 300
)
