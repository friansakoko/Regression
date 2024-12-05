library(quantmod)
library(ggplot2)

# Load stock data (e.g., Apple) from Yahoo Finance
getSymbols("AAPL", src = "yahoo", from = "2018-01-01", to = "2023-12-01")

# Build dataframe----
# Convert to a data frame with Date included
model_data <- data.frame(
  Date = index(AAPL),  # Extract dates
  coredata(AAPL)       # Extract data columns
)

View(model_data)

# Rename columns for simplicity
colnames(model_data) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")

# Check the structure of the dataset
str(model_data)

# Remove missing values (if any)
model_data <- na.omit(model_data)

# Split data into training (80%) and testing (20%)
set.seed(42)  # For reproducibility
train_indices <- sample(1:nrow(model_data), size = 0.8 * nrow(model_data))
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

# Build a linear regression model----
# Model(Target ~ Predictor 1 + Predictor 2 + ...)
lm_model <- lm(Adjusted ~ Open + High + Low + Close + Volume, data = train_data) # poly(Predictor1, 2) + poly(Predictor2, 2) + Predictor3

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
  Actual = test_data$Adjusted,
  Predicted = test_data$Predicted
)
head(results)

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
# Plot Actual vs Predicted
plot(results$Actual, results$Predicted,
     main = "Actual vs Predicted Adjusted Close",
     xlab = "Actual Adjusted Close",
     ylab = "Predicted Adjusted Close",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # Add a diagonal reference line
grid()

# Create a ggplot to visualize testing predictions vs. actuals over time
ggplot(test_data, aes(x = Date)) +
  geom_line(aes(y = Adjusted, color = "Actual"), linewidth = 1) +  # Actual values
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1) +  # Predicted values
  labs(title = "Testing Predictions vs Actuals Over Time",
       x = "Date", y = "Adjusted Close") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Model Equation----
coefficients <- coef(lm_model)
intercept <- coefficients[1]
slopes <- coefficients[-1]

equation <- paste0("Adjusted = ", round(intercept, 4), 
                   " + ", paste0(round(slopes, 4), " * ", names(slopes), collapse = " + "))
cat("Regression Equation of Linear Regression:\n")
cat(equation, "\n")

# Testing the model, into new data----
# Load new data for January to February 2024
new_data <- getSymbols("AAPL", src = "yahoo", from = "2024-01-01", to = "2024-02-29", auto.assign = FALSE)

# Convert to data frame with Date included
new_data <- data.frame(
  Date = index(new_data),
  coredata(new_data)
)

# Rename columns for consistency
colnames(new_data) <- c("Date", "Open", "High", "Low", "Close", "Volume", "Adjusted")

# Check the structure of the new data
str(new_data)

# Make predictions using the existing model
new_data$Predicted <- predict(lm_model, newdata = new_data)

# Check predictions
head(new_data)

# Plot predictions for the new data
ggplot(new_data, aes(x = Date)) +
  geom_line(aes(y = Adjusted, color = "Actual"), linewidth = 1) +  # Actual values
  geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1) +  # Predicted values
  labs(title = "Predictions for New Data (Jan-Feb 2024)",
       x = "Date", y = "Adjusted Close") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Calculate Mean Squared Error (MSE) for the new prediction----
# Calculate the Mean Squared Error (MSE)
mse <- mean((new_data$Adjusted - new_data$Predicted)^2, na.rm = TRUE)

# Print the MSE
print(paste("Mean Squared Error:", round(mse, 4)))


# Calculate Rsquare for the new prediction----
# Calculate Total Sum of Squares (SS_tot)
ss_tot <- sum((new_data$Adjusted - mean(new_data$Adjusted))^2, na.rm = TRUE)

# Calculate Residual Sum of Squares (SS_res)
ss_res <- sum((new_data$Adjusted - new_data$Predicted)^2, na.rm = TRUE)

# Calculate R-squared
r_squared <- 1 - (ss_res / ss_tot)

# Print the R-squared value
print(paste("R-squared:", round(r_squared, 4)))
