function RMSE = compute_RMSE(y_obtained, y_true)
    n_samples = length(y_obtained);
    RMSE = sqrt(sum((y_obtained-y_true).^2)/n_samples);
    
end

