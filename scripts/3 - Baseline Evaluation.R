install.packages("remotes")
install.packages("tidyverse")

remotes::install_github("b0rxa/scmamp", force=TRUE)
BiocManager::install("Rgraphviz")

library(tidyverse)
library(scmamp)

df <- read_csv('experiment_results.csv')

df_pivot <- df |> 
  filter(Preprocessor == "Baseline") |> 
  pivot_wider(
    names_from = Model,
    values_from = `F1 Macro`,
    id_cols = Dataset
  ) |> 
  select(-Dataset)

################################################################################
# Simulação de Dados Futuros
set.seed(42)
model_names <- colnames(df_pivot)

simulated_data <- map_df(1:17, function(i) {
  base_performance <- runif(length(model_names), 0.70, 0.85)
  names(base_performance) <- model_names
  
  performance <- base_performance + c(0.10, 0.02, 0.08, -0.05, 0.01, -0.15)
  performance <- pmin(pmax(performance, 0), 1)
  performance <- performance + runif(length(model_names), -0.02, 0.02)
})

df_pivot <- bind_rows(df_pivot, simulated_data)
################################################################################

imanDavenportTest(df_pivot)

test.res <- postHocTest(data = df_pivot, test = 'friedman', correct = 'bergmann', use.rank = TRUE)

plotRanking(pvalues=test.res$corrected.pval, summary=test.res$summary)