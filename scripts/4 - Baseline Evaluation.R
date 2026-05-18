install.packages("tidyverse")
remotes::install_github("b0rxa/scmamp", force=TRUE)

library(tidyverse)
library(scmamp)

results_df <- read_csv('../data/aggregated_results.csv')

baseline_df <- results_df |> 
  filter(Preprocessor == "Baseline") |> 
  select(Dataset, Model, Test_MCC) |> 
  pivot_wider(names_from = Model, values_from = Test_MCC)

baseline_matrix <- baseline_df |> select(-Dataset)

baseline_posthoc <- postHocTest(baseline_matrix, test = 'friedman', correct = 'bergmann', use.rank = TRUE)

png("../images/algorithms_baseline.png", width = 1000, height = 400, res = 120)
plotRanking(pvalues = baseline_posthoc$corrected.pval, summary = baseline_posthoc$summary)
dev.off()

scaled_df <- results_df |> 
  filter(Preprocessor == "Standard Scaling") |> 
  select(Dataset, Model, Test_MCC) |> 
  pivot_wider(names_from = Model, values_from = Test_MCC)

scaled_matrix <- scaled_df |> select(-Dataset)

scaled_posthoc <- postHocTest(scaled_matrix, test = 'friedman', correct = 'bergmann', use.rank = TRUE)

png("../images/algorithms_scaled.png", width = 1000, height = 400, res = 120)
plotRanking(pvalues = scaled_posthoc$corrected.pval, summary = scaled_posthoc$summary)
dev.off()