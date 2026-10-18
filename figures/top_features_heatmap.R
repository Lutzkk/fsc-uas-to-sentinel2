library(ggplot2)
library(tidyverse)
library(dplyr)

# --- Paths --- (relative to repo root, run: Rscript figures/top_features_heatmap.R)
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

# --- Band name lookup ---
band_labels <- c(
  "B2"   = "Blue",
  "B3"   = "Green",
  "B4"   = "Red",
  "B5"   = "Red Edge 1",
  "B6"   = "Red Edge 2",
  "B7"   = "Red Edge 3",
  "B8"   = "NIR",
  "B8A"  = "Narrow NIR",
  "B11"  = "SWIR-1",
  "B12"  = "SWIR-2",
  "NDSI" = "NDSI",
  "NDVI" = "NDVI",
  "NDWI" = "NDWI",
  "DEM"  = "DEM"
)

# --- Feature frequency in top 5% models ---
TOP_PROP <- 0.05

top_features_pct <- results %>%
  group_by(model) %>%
  slice_min(RMSE, prop = TOP_PROP) %>%
  ungroup() %>%
  unnest(features) %>%
  count(model, features, name = "n") %>%
  group_by(model) %>%
  mutate(pct = n / sum(n) * 100) %>%
  ungroup() %>%
  complete(model, features, fill = list(pct = 0, n = 0)) %>%
  mutate(features = recode(features, !!!band_labels))

keep_features <- top_features_pct %>%
  group_by(model) %>%
  arrange(desc(pct), .by_group = TRUE) %>%
  mutate(rank = row_number(), total = n()) %>%
  filter(rank <= 3 | rank > total - 3) %>%
  ungroup() %>%
  distinct(features) %>%
  pull(features)

top_features_pct <- top_features_pct %>%
  filter(features %in% keep_features)

feat_order <- top_features_pct %>%
  filter(model == "SVR") %>%
  arrange(pct) %>%
  pull(features)

# --- Plot ---
top_features_pct %>%
  mutate(features = factor(features, levels = feat_order)) %>%
  ggplot(aes(x = model, y = features, fill = pct)) +
  geom_tile(color = "white", linewidth = 1, width = 0.8) +
  geom_text(aes(label = paste0(round(pct), "%")),
            size = 8, fontface = "bold", color = "black") +
  scale_fill_gradient(low = "#d6e4f0", high = "#2166AC") +
  scale_y_discrete(expand = expansion(0)) +
  scale_x_discrete(expand = expansion(0), position = "top") +
  labs(
    title    = "Feature Occurrence in Top 5% Models — RF vs SVR",
    subtitle = "Most and least used features per model | sorted by SVR occurrence",
    x = NULL, y = NULL
  ) +
  guides(fill = "none") +
  theme_minimal(base_size = 24) +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title    = element_text(hjust = 0.5, size = 26, face = "bold", color = "black"),
    plot.subtitle = element_text(hjust = 0.5, size = 18, color = "black"),
    axis.text.x   = element_text(size = 28, face = "bold", color = "black"),
    axis.text.y   = element_text(size = 20, face = "bold", color = "black",
                                 hjust = 1, margin = margin(r = -8)),
    panel.grid    = element_blank(),
    plot.margin   = margin(6, 6, 6, 2)
  )

ggsave(
  "results/plots/top_features_heatmap.png",
  width = 16, height = 9, dpi = 300
)
