# Load the data from the URL
url <- "https://raw.githubusercontent.com/friansakoko/Regression/refs/heads/main/housing_price_dataset.csv"
model_data <- read.csv(url)
View(model_data)

library(ggplot2)

# Remove missing values (if any)
# model_data <- na.omit(model_data)

# Split data into training (80%) and testing (20%)
set.seed(42)  # For reproducibility
train_indices <- sample(1:nrow(model_data), size = 0.8 * nrow(model_data))
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

# Build a linear regression model----
lm_model <- lm(Price ~ SquareFeet + Bedrooms + Bathrooms + YearBuilt, data = train_data) # (Target ~ poly(Predictor1, 2) + poly(Predictor2, 2) + Predictor3)

# Display the model summary
summary(lm_model)

# Testing----
# Make predictions on the testing dataset
test_data$Predicted <- predict(lm_model, newdata = test_data)

# Check the first few rows of the test data with predictions
head(test_data)

# Model Evaluation----
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Price,
  Predicted = test_data$Predicted
)
head(results)

str(results)

# Calculate Mean Squared Error (MSE)
mse <- mean((results$Actual - results$Predicted)^2)
print(paste("MSE of Linear Regression:", round(mse, 4)))

# Calculate RSquare
# Calculate Total Sum of Squares (SS_tot)
ss_tot <- sum((results$Actual - mean(results$Actual))^2, na.rm = TRUE)

# Calculate Residual Sum of Squares (SS_res)
ss_res <- sum((results$Actual - results$Predicted)^2, na.rm = TRUE)

# Calculate R-squared
r_squared <- 1 - (ss_res / ss_tot)

# Print the R-squared value
print(paste("R-squared of Linear Regression:", round(r_squared, 4)))


# Visualization----

options(scipen = 999)  # Disable scientific notation
plot(results$Actual, results$Predicted,
     main = "Actual vs Predicted Price",
     xlab = "Actual Price ($)",
     ylab = "Predicted Price ($)",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # Add a diagonal reference line
grid()

# Create a ggplot to visualize testing predictions vs. actuals over time
ggplot(test_data, aes(x = YearBuilt)) +
  geom_point(aes(y = Price, color = "Actual"), size = 1) +  # Actual values
  geom_point(aes(y = Predicted, color = "Predicted"), size = 1) +  # Predicted values
  labs(title = "Testing Predictions vs Actuals Over Time",
       x = "Year", y = "Predicted Price ($)") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Model Equation----
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

# Format the equation
equation <- paste0("Price = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

# Testing the model, into new data----
# Assume lm_model is your trained linear regression model
# Example: lm_model <- lm(Price ~ SquareFeet + Bedrooms + Bathrooms + YearBuilt, data = training_data)

# Prompt user to input new data
cat("Please add new data:\nSquareFeet, Bedrooms, Bathrooms, YearBuilt:\n")
input_data <- as.numeric(strsplit(readline(), ",")[[1]])

# Convert input data to a dataframe
new_data <- data.frame(
  SquareFeet = input_data[1],
  Bedrooms = input_data[2],
  Bathrooms = input_data[3],
  YearBuilt = input_data[4]
)

# Predict using the model
predicted_price <- predict(lm_model, newdata = new_data)

# Show the result
cat("Predicted Price ($): ", predicted_price, "\n")
