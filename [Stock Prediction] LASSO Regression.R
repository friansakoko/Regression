library(quantmod)
library(glmnet)
library(ggplot2)

# Load stock data (e.g., Apple) from Yahoo Finance
getSymbols("AAPL", src = "yahoo", from = "2018-01-01", to = "2023-12-01")

# Build dataframe----
# Convert to a data frame with Date included
model_data <- data.frame(
  Date = index(AAPL),  # Extract dates
  coredata(AAPL)       # Extract data columns
)

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

# Prepare predictors and target for training and testing
x_train <- as.matrix(train_data[, c("Open", "High", "Low", "Close", "Volume")])
y_train <- train_data$Adjusted
x_test <- as.matrix(test_data[, c("Open", "High", "Low", "Close", "Volume")])
y_test <- test_data$Adjusted

# Build a LASSO regression model----
# Fit LASSO regression model
# alpha = 1 for LASSO (default), lambda is the regularization parameter
lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1  # 1 = LASSO regression
)

# Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  nfolds = 10  # Number of folds for cross-validation
)

# Best lambda from cross-validation
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Fit the final LASSO model with the optimal lambda
final_lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# Display the model summary
summary(final_lasso_model)

# Testing----
# Predict on test data using the optimal lambda
# Make predictions on the test set
test_data$Predicted_lasso <- predict(final_lasso_model, s = best_lambda, newx = x_test)

# Explicitly assign the prediction results to a new column with the desired name
test_data$Prediction <- as.numeric(predict(final_lasso_model, s = best_lambda, newx = x_test))

# Check the first few rows to confirm the column name change
head(test_data)

# Check the first few rows with predictions
head(test_data)

# Model Evaluation----
# Compare actual vs. predicted values
results <- data.frame(
  Actual = test_data$Adjusted,
  Predicted = test_data$Predicted_lasso
)
head(results)

# Mean Squared Error (MSE)
mse_lasso <- mean((y_test - test_data$Predicted_lasso)^2)
print(paste("Mean Squared Error (Lasso):", round(mse_lasso, 4)))

# R-squared
ss_tot_lasso <- sum((y_test - mean(y_test))^2)
ss_res_lasso <- sum((y_test - test_data$Predicted_lasso)^2)
r_squared_lasso <- 1 - (ss_res_lasso / ss_tot_lasso)
print(paste("R-squared (Lasso):", round(r_squared_lasso, 4)))

# Visualization----
# Visualize Actual vs Predicted (Lasso)
# Plot Actual vs Predicted
plot(results$Actual, results$Predicted,
     main = "Actual vs Predicted Adjusted Close",
     xlab = "Actual Adjusted Close",
     ylab = "Predicted Adjusted Close",
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # Add a diagonal reference line

# Create a ggplot to visualize testing predictions vs. actuals over time
ggplot(test_data, aes(x = Date)) +
  geom_line(aes(y = Adjusted, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted_lasso, color = "Predicted"), size = 1) +
  labs(title = "Lasso Regression: Actual vs Predicted Adjusted Close",
       x = "Date", y = "Adjusted Close") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# Model Equation----
# Extract coefficients at the optimal lambda
coefficients <- coef(final_lasso_model, s = best_lambda)

# Convert to a named vector for easier access
coefficients <- as.vector(coefficients)
names(coefficients) <- rownames(coef(lasso_model, s = best_lambda))

# Intercept
intercept <- coefficients[1]

# Coefficients for predictors
slopes <- coefficients[-1]

# Format the equation
equation <- paste0(
  "Adjusted = ", round(intercept, 4), " + ",
  paste0(round(slopes, 4), " * ", names(slopes), collapse = " + ")
)

# Display the equation
cat("Lasso Regression Model Equation:\n")
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

# Ensure there are no missing values in the new data
new_data <- na.omit(new_data)

# Prepare predictors for the new data
x_new <- as.matrix(new_data[, c("Open", "High", "Low", "Close", "Volume")])

# Make predictions using the existing Lasso Regression model
new_data$Predicted <- predict(final_lasso_model, s = best_lambda, newx = x_new)

# Display the first few rows of the new data with predictions
head(new_data)

# Plot predictions for the new data
ggplot(new_data, aes(x = Date)) +
  geom_line(aes(y = Adjusted, color = "Actual"), size = 1) +  # Actual values
  geom_line(aes(y = Predicted, color = "Predicted"), size = 1) +  # Predicted values
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
