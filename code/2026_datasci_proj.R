
# Code ran in sapello
 
# shared_lib_path <- "/work/crss8030/instructor_data/shared_R_libs"

#class_packages <- c("tidymodels", "tidyverse", "vip", "ranger", "finetune", "parsnip", "reticulate", "xgboost", "doParallel", "lme4", "here")

#install.packages(class_packages, lib = shared_lib_path)

# .libPaths(c("/work/crss8030/instructor_data/shared_R_libs", .libPaths()))

#install.packages("xgboost") #new pacakage

library(tidymodels)
library(finetune)
library(vip)
library(xgboost)
library(tidyverse)
library(doParallel)
library(here)
library(earth)


train_model_ready <- read_csv(here("data", "train_model_ready.csv")) %>%
  select(-any_of(c("...1", "n_plots", "hybrid", "previous_crop", "inbred"))) %>%
  mutate(
    year = as.numeric(year),
    site = as.character(site)
  )

test_model_ready <- read_csv(here("data", "test_model_ready.csv")) %>%
  select(-any_of(c("...1", "n_plots", "hybrid", "previous_crop", "inbred"))) %>%
  mutate(
    year = as.numeric(year),
    site = as.character(site)
  )

set.seed(931735)

train_data <- train_model_ready %>%
  filter(year <= 2022)

valid_data_2023 <- train_model_ready %>%
  filter(year == 2023)

data_split_2023 <- make_splits(
  list(
    analysis = which(train_model_ready$year <= 2022),
    assessment = which(train_model_ready$year == 2023)
  ),
  data = train_model_ready
)

density_plot <- ggplot() +
  geom_density(data = train_data,
               aes(x = yield_mg_ha),
               color = "red") +
  geom_density(data = valid_data_2023,
               aes(x = yield_mg_ha),
               color = "blue") +
  labs(
    x = "Yield (Mg/ha)",
    y = "Density",
    title = "Distribution of Yield: Training vs 2023 Validation"
  )

ggsave(
  plot = density_plot,
  filename = here("output", "png", "density_plot_train_vs_2023.png"),
  height = 6,
  width = 9,
  dpi = 600
)


weather_recipe <- recipe(yield_mg_ha ~ ., data = train_data) %>%
  step_rm(year) %>% #Remove the column year from the predictors before modeling
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

# Prep the recipe to estimate any required statistics
weather_prep <- weather_recipe %>%
  prep()

set.seed(235)
# Setting up cros validations in folds, we used 10 in class. we can use 10 in Sapelo but with big grid_hypercuvbe size
resampling_foldcv <- vfold_cv(
  train_data,
  v = 10,
  strata = yield_mg_ha
)

try(stopCluster(cl), silent = TRUE)
stopImplicitCluster()
registerDoSEQ()
closeAllConnections()
gc()

n_cores <- as.numeric(Sys.getenv("SLURM_CPUS_ON_NODE"))

if (is.na(n_cores)) {
  n_cores <- max(1, parallel::detectCores() - 1)
}

cl <- makePSOCKcluster(n_cores)
registerDoParallel(cl)

cat(paste0("\nRegistered ", n_cores, " cores\n"))

xgb_spec <- boost_tree(
  # Total number of boosting itinerations
  trees = tune(),
  # Maximum depth of each tree
  tree_depth = tune(),
  # Minimum samples required to split a node
  min_n = tune(),
  # Step size shrinkage for each boosting step
  learn_rate = tune()
) %>%
  # Specify engine
  set_engine("xgboost") %>%
  # Specify mode (Regression, classification)
  set_mode("regression")

set.seed(123)

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  learn_rate(),
  trees(),
  size = 50
)


xgb_grid

xgb_grid_plot <- ggplot(
  data = xgb_grid,
  aes(x = tree_depth, y = min_n)
) +
  geom_point(
    aes(color = factor(learn_rate), size = trees),
    alpha = 0.5,
    show.legend = FALSE
  ) +
  labs(title = "XGBoost Hyperparameter Grid")

ggsave(
  plot = xgb_grid_plot,
  filename = here("output", "png", "xgb_hyperparameter_grid.png"),
  height = 6,
  width = 9,
  dpi = 600
)

set.seed(76544)

xgb_res <- tune_race_anova(
  object = xgb_spec,
  preprocessor = weather_recipe,
  resamples = resampling_foldcv,
  grid = xgb_grid,
  metrics = metric_set(rmse, rsq, mae),
  control = control_race(
    save_pred = TRUE,
    verbose = TRUE,
    parallel_over = "resamples"
  )
)

saveRDS(xgb_res, here("output", "rds", "xgb_res.rds"))

xgb_metrics <- collect_metrics(xgb_res)

write_csv(
  xgb_metrics,
  here("output", "csv", "xgb_tuning_metrics.csv")
)

best_xgb <- select_best(xgb_res, metric = "rmse")

write_csv(
  best_xgb,
  here("output", "csv", "best_xgb_hyperparameters.csv")
)


final_xgb_workflow <- workflow() %>%
  add_recipe(weather_recipe) %>%
  add_model(xgb_spec) %>%
  finalize_workflow(best_xgb)

xgb_train_fit <- final_xgb_workflow %>%
  fit(data = train_data)

xgb_train_pred <- predict(
  xgb_train_fit,
  new_data = train_data
) %>%
  bind_cols(train_data)

write_csv(
  xgb_train_pred,
  here("output", "csv", "xgb_train_pred.csv")
)

xgb_train_metrics <- xgb_train_pred %>%
  metrics(
    truth = yield_mg_ha,
    estimate = .pred
  ) %>%
  mutate(estimate = round(.estimate, 3))

xgb_train_metrics

write_csv(
  xgb_train_metrics,
  here("output", "csv", "xgb_training_metrics.csv")
)

set.seed(10)

xgb_fit_2023 <- last_fit(
  final_xgb_workflow,
  split = data_split_2023,
  metrics = metric_set(rmse, rsq, mae)
)

xgb_test_met <- collect_metrics(xgb_fit_2023) %>%
  mutate(estimate = round(.estimate, 3))

xgb_pred_2023 <- collect_predictions(xgb_fit_2023)

write_csv(
  xgb_test_met,
  here("output", "csv", "xgb_2023_validation_metrics.csv")
)

write_csv(
  xgb_pred_2023,
  here("output", "csv", "xgb_2023_validation_predictions.csv")
)

xgb_publication_ready <- xgb_train_pred %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(
    aes(fill = yield_mg_ha),
    shape = 21,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(color = "red", linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)",
    title = "XGBoost: Observed vs Predicted Yield"
  ) +
  annotate(
    "label",
    x = Inf,
    y = -Inf,
    label = paste0(
      "R-sq: ",
      xgb_train_metrics$estimate[xgb_train_metrics$.metric == "rsq"],
      "\nRMSE: ",
      xgb_train_metrics$estimate[xgb_train_metrics$.metric == "rmse"]
    ),
    hjust = 1.1,
    vjust = -0.5
  ) +
  theme(
    panel.background = element_rect(fill = "gray82"),
    panel.grid = element_blank()
  )

ggsave(
  plot = xgb_publication_ready,
  filename = here("output", "png", "xgb_predicted_vs_observed_2023.png"),
  height = 6,
  width = 9,
  dpi = 600
)

str(xgb_pred_test$yield_mg_ha)
summary(xgb_pred_test$yield_mg_ha)

xgb_vip <- final_xgb_workflow %>%
  fit(data = train_data) %>%
  extract_fit_parsnip() %>%
  vip::vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  slice_max(Importance, n = 25) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Importance)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(
    y = NULL,
    title = "XGBoost Variable Importance"
  ) +
  theme(
    panel.background = element_rect(fill = "gray82"),
    panel.grid = element_blank()
  )

ggsave(
  plot = xgb_vip,
  filename = here("output", "png", "xgb_variable_importance.png"),
  height = 6,
  width = 9,
  dpi = 600
)

final_xgb_recipe_full <- recipe(yield_mg_ha ~ ., data = train_model_ready) %>%
  step_rm(year) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

final_xgb_workflow_full <- workflow() %>%
  add_recipe(final_xgb_recipe_full) %>%
  add_model(xgb_spec) %>%
  finalize_workflow(best_xgb)

set.seed(10)

final_xgb_fit <- final_xgb_workflow_full %>%
  fit(data = train_model_ready)

xgb_pred_test <- predict(
  final_xgb_fit,
  new_data = test_model_ready
) %>%
  bind_cols(test_model_ready) %>%
  rename(predicted_yield_mg_ha = .pred)

write_csv(
  xgb_pred_test,
  here("output", "csv", "xgb_test_model_ready_predictions.csv")
)

saveRDS(
  final_xgb_fit,
  here("output", "rds", "final_xgb_fit.rds")
)


mars_spec <- mars(
  num_terms = tune(),
  prod_degree = tune(),
  prune_method = tune()
) %>%
  set_engine("earth") %>%
  set_mode("regression")


mars_workflow <- workflow() %>%
  add_recipe(weather_recipe) %>%
  add_model(mars_spec)


mars_grid <- grid_regular(
  num_terms(range = c(5L, 50L)),
  prod_degree(range = c(1L, 2L)),
  prune_method(values = c("backward", "none")),
  levels = c(
    num_terms = 5,
    prod_degree = 2,
    prune_method = 2
  )
)

set.seed(1234)

mars_res <- tune_grid(
  mars_workflow,
  resamples = resampling_foldcv,
  grid = mars_grid,
  metrics = metric_set(rmse, rsq, mae),
  control = control_grid(
    save_pred = TRUE,
    verbose = TRUE,
    parallel_over = "resamples"
  )
)

saveRDS(mars_res, here("output", "rds", "mars_res.rds"))

mars_metrics <- collect_metrics(mars_res)

write_csv(
  mars_metrics,
  here("output", "csv", "mars_tuning_metrics.csv")
)

best_mars <- select_best(mars_res, metric = "rmse")

write_csv(
  best_mars,
  here("output", "csv", "best_mars_hyperparameters.csv")
)


final_mars_workflow <- mars_workflow %>%
  finalize_workflow(best_mars)


mars_train_fit <- final_mars_workflow %>%
  fit(data = train_data)

mars_train_pred <- predict(
  mars_train_fit,
  new_data = train_data
) %>%
  bind_cols(train_data)

write_csv(
  mars_train_pred,
  here("output", "csv", "mars_train_pred.csv")
)

mars_train_metrics <- mars_train_pred %>%
  metrics(
    truth = yield_mg_ha,
    estimate = .pred
  ) %>%
  mutate(estimate = round(.estimate, 3))

mars_train_metrics

write_csv(
  mars_train_metrics,
  here("output", "csv", "mars_training_metrics.csv")
)

set.seed(10)

mars_fit_2023 <- last_fit(
  final_mars_workflow,
  split = data_split_2023,
  metrics = metric_set(rmse, rsq, mae)
)

mars_test_met <- collect_metrics(mars_fit_2023) %>%
  mutate(estimate = round(.estimate, 3))

mars_pred_2023 <- collect_predictions(mars_fit_2023)

write_csv(
  mars_test_met,
  here("output", "csv", "mars_2023_validation_metrics.csv")
)

write_csv(
  mars_pred_2023,
  here("output", "csv", "mars_2023_validation_predictions.csv")
)

mars_publication_ready <- mars_train_pred %>%
  ggplot(aes(x = yield_mg_ha, y = .pred)) +
  geom_point(
    aes(fill = yield_mg_ha),
    shape = 21,
    alpha = 0.7,
    show.legend = FALSE
  ) +
  scale_fill_viridis_c(option = "H") +
  geom_abline(color = "red", linetype = 2) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Observed Yield (Mg/ha)",
    y = "Predicted Yield (Mg/ha)",
    title = "MARS: Observed vs Predicted Yield"
  ) +
  annotate(
    "label",
    x = Inf,
    y = -Inf,
    label = paste0(
      "R-sq: ",
      mars_train_metrics$estimate[mars_train_metrics$.metric == "rsq"],
      "\nRMSE: ",
      mars_train_metrics$estimate[mars_train_metrics$.metric == "rmse"]
    ),
    hjust = 1.1,
    vjust = -0.5
  ) +
  theme(
    panel.background = element_rect(fill = "gray82"),
    panel.grid = element_blank()
  )

ggsave(
  plot = mars_publication_ready,
  filename = here("output", "png", "mars_predicted_vs_observed_2023.png"),
  height = 6,
  width = 9,
  dpi = 600
)

final_mars_recipe_full <- recipe(yield_mg_ha ~ ., data = train_model_ready) %>%
  step_rm(year) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

final_mars_workflow_full <- workflow() %>%
  add_recipe(final_mars_recipe_full) %>%
  add_model(mars_spec) %>%
  finalize_workflow(best_mars)

set.seed(10)

final_mars_fit <- final_mars_workflow_full %>%
  fit(data = train_model_ready)

mars_pred_test <- predict(
  final_mars_fit,
  new_data = test_model_ready
) %>%
  bind_cols(test_model_ready) %>%
  rename(predicted_yield_mg_ha = .pred)

write_csv(
  mars_pred_test,
  here("output", "csv", "mars_test_model_ready_predictions.csv")
)

saveRDS(
  final_mars_fit,
  here("output", "rds", "final_mars_fit.rds")
)


mars_vip <- final_mars_workflow %>%
  fit(data = train_data) %>%
  extract_fit_parsnip() %>%
  vip::vi() %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  slice_max(Importance, n = 25) %>%
  ggplot(aes(x = Importance, y = Variable, fill = Importance)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(
    y = NULL,
    title = "MARS Variable Importance"
  ) +
  theme(
    panel.background = element_rect(fill = "gray82"),
    panel.grid = element_blank()
  )

mars_vip

ggsave(
  plot = mars_vip,
  filename = here("output", "png", "mars_variable_importance.png"),
  height = 6,
  width = 9,
  dpi = 600
)

stopCluster(cl)
registerDoSEQ()
closeAllConnections()
gc()
knitr::purl("2026_only_ml_sapello.qmd", output = "sapelo_script.R", documentation = 0)

