
library(tidyverse)
library(plotly)
library(png)
library(grid)
library(maps)
library(shiny)

# Load df_model
df_model <- df_model <- read_csv("data/train_model_ready.csv")

df_traits <- read_csv("data/training/training_meta_filtered.csv") 

df_train_predictions_xgb <-read_csv("data/xgb_train_pred.csv")

df_train_predictions_mars <-read_csv("data/mars_train_pred.csv")

names(df_model)

# Define UI
ui <- fluidPage(
  titlePanel("Corn Yield Trials"),
  
  tags$head(
    tags$style(HTML("
      /* Center the tabs */
      .nav-tabs {
        display: flex;
        justify-content: center;
      }

      /* Style all tab titles */
      .nav-tabs > li > a {
        font-weight: bold;
        color: black !important;
        text-align: center;
      }

      /* Active tab styling */
      .nav-tabs > li.active > a {
        font-weight: bold;
        color: black !important;
        background-color: #e6e6e6 !important;
        border-radius: 5px;
      }
    "))
  ),
  
  tabsetPanel(
    id = "tabs",
    
    tabPanel(
      "Experiment Sites",
      plotOutput("sites_map", height = "900px", width = "900px")
    ),
    
    tabPanel(
      "Exploratory Data Analysis Variables (Yield x Variable)",
      selectInput(
        "group_var",
        "Select Variable:",
        choices = names(df_model)[!(names(df_model) %in% c("yield_mg_ha", "hybrid"))],
        selected = "previous_crop"
      ),
      plotlyOutput("boxplot")
    ),
    
    tabPanel(
      "Climate Variables Density Plot",
      
      fluidRow(
        column(
          3,
          selectInput(
            "clim_var1",
            "Select Variable for Plot 1:",
            choices = names(df_model)[!(names(df_model) %in% c(
              "...1", "year", "site", "hybrid", "site_original",
              "yield_mg_ha", "n_plots", "previous_crop", "soilpH","om_pct", "soilk_ppm", "soilp_ppm"
            ))],
            selected = "prcp_total"
          )
        ),
        
        column(
          3,
          selectInput(
            "clim_var2",
            "Select Variable for Plot 2:",
            choices = names(df_model)[!(names(df_model) %in% c(
              "...1", "year", "site", "hybrid", "site_original",
              "yield_mg_ha", "n_plots", "previous_crop","om_pct", "soilk_ppm", "soilp_ppm"
            ))],
            selected = "tmax_mean"
          )
        ),
        
        column(
          3,
          selectInput(
            "clim_var3",
            "Select Variable for Plot 3:",
            choices = names(df_model)[!(names(df_model) %in% c(
              "...1", "year", "site", "hybrid", "site_original",
              "yield_mg_ha", "n_plots", "previous_crop","om_pct", "soilk_ppm", "soilp_ppm"
            ))],
            selected = "tmin_mean"
          )
        ),
        
        column(
          3,
          selectInput(
            "clim_var4",
            "Select Variable for Plot 4:",
            choices = names(df_model)[!(names(df_model) %in% c(
              "...1", "year", "site", "hybrid", "site_original",
              "yield_mg_ha", "n_plots", "previous_crop","om_pct", "soilk_ppm", "soilp_ppm"
            ))],
            selected = "srad_mean"
          )
        )
      ),
      
      fluidRow(
        column(6, plotlyOutput("density_plot1")),
        column(6, plotlyOutput("density_plot2"))
      ),
      
      fluidRow(
        column(6, plotlyOutput("density_plot3")),
        column(6, plotlyOutput("density_plot4"))
      )
    ),
    
    tabPanel(
      "Yield Density Plot",
      plotlyOutput("density_plot")
    ),
    
    tabPanel(
      "Variable Importance Based on Model",
      fluidRow(
        column(
          6,
          uiOutput("xgb_vi_plot")
        ),
        column(
          6,
          uiOutput("mars_vi_plot")
        )
      )
    ),
    
    tabPanel(
      "Predicted vs Actual for Training Dataset",
      fluidRow(
        column(
          6,
          uiOutput("XGBoost")
        ),
        column(
          6,
          uiOutput("MARS")
        )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  # US map
  output$sites_map <- renderPlot({
    
    states <- map_data("state")
    
    ggplot() +
      geom_polygon(
        data = states,
        aes(x = long, y = lat, group = group),
        fill = "gray95",
        color = "black"
      ) +
      geom_point(
        data = df_traits,
        aes(
          x = longitude,
          y = latitude,
          color = site
        ),
        size = 3
      ) +
      coord_fixed(1.3) +
      theme_minimal() +
      labs(
        title = "Experiment Sites",
        x = "Longitude",
        y = "Latitude",
        color = "Site"
      ) +
      theme(
        plot.title = element_text(size = 23, face = "bold"),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 18),
        axis.title.x = element_text(size = 18),
        axis.title.y = element_text(size = 18),
        axis.text.x = element_text(size = 14),
        axis.text.y = element_text(size = 14)
      )
  })

  # Boxplot
  output$boxplot <- renderPlotly({
    filtered_data <- df_model %>%
      filter(!is.na(!!sym(input$group_var)), !is.na(yield_mg_ha))
    filtered_data[[input$group_var]] <- as.factor(filtered_data[[input$group_var]])
    p <- ggplot(filtered_data, aes_string(x = input$group_var, y = "yield_mg_ha")) +
      geom_boxplot() +
      labs(
        title = paste("Boxplot of Yield by", input$group_var),
        x = input$group_var,
        y = "Yield (mg/ha)"
      ) +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
    ggplotly(p)
  })
  
  #Density for climate variables
  
  output$density_plot1 <- renderPlotly({
    p <- ggplot(df_model, aes(x = .data[[input$clim_var1]])) +
      geom_density(fill = "blue", alpha = 0.6) +
      theme_minimal() +
      labs(
        title = paste("Density Plot of", input$clim_var1),
        x = input$clim_var1,
        y = "Density"
      )
    
    ggplotly(p)
  })
  
  output$density_plot2 <- renderPlotly({
    p <- ggplot(df_model, aes(x = .data[[input$clim_var2]])) +
      geom_density(fill = "green", alpha = 0.6) +
      theme_minimal() +
      labs(
        title = paste("Density Plot of", input$clim_var2),
        x = input$clim_var2,
        y = "Density"
      )
    
    ggplotly(p)
  })
  
  output$density_plot3 <- renderPlotly({
    p <- ggplot(df_model, aes(x = .data[[input$clim_var3]])) +
      geom_density(fill = "red", alpha = 0.6) +
      theme_minimal() +
      labs(
        title = paste("Density Plot of", input$clim_var3),
        x = input$clim_var3,
        y = "Density"
      )
    
    ggplotly(p)
  })
  
  output$density_plot4 <- renderPlotly({
    p <- ggplot(df_model, aes(x = .data[[input$clim_var4]])) +
      geom_density(fill = "pink", alpha = 0.6) +
      theme_minimal() +
      labs(
        title = paste("Density Plot of", input$clim_var4),
        x = input$clim_var4,
        y = "Density"
      )
    
    ggplotly(p)
  })
  
  # Density Plot
  output$density_plot <- renderPlotly({
    p <- ggplot(df_model, aes(x = yield_mg_ha)) +
      geom_density(fill = "red", alpha = 0.5) +
      labs(
        title = "Density Plot of Yield",
        x = "Yield (mg/ha)",
        y = "Density"
      ) +
      theme_minimal()
    ggplotly(p)
  })
  
  # Variable Importance Plot
  output$xgb_vi_plot <- renderUI({
    tags$img(
      src = "xgb_variable_importance.png",
      width = "100%"
    )
  })
  
  output$mars_vi_plot <- renderUI({
    tags$img(
      src = "mars_variable_importance.png",
      width = "100%"
    )
  })
  
  
  
  # Results: Predicted vs Actual Plot
  output$XGBoost <- renderUI({
    tags$img(
      src = "xgb_predicted_vs_observed_2023.png",
      width = "100%"
    )
  })
  
  output$MARS <- renderUI({
    tags$img(
      src = "mars_predicted_vs_observed_2023.png",
      width = "100%"
    )
  })
}
# Run the App
shinyApp(ui = ui, server = server)

