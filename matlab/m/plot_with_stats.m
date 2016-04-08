function plot_with_stats(data)

  hold off;

  plot(data, 'k-+', 'color', [0.3 0.3 0.3]); hold on;

  n = length(data);
  plot(1:n, repmat(mean(data), 1, n), 'k--', 'linewidth', 2);
  plot(1:n, repmat(median(data), 1, n), 'k-', 'linewidth', 3);

  legend('data', sprintf('mean %0.2f', mean(data)), ...
    sprintf('median %0.2f', median(data)), 'location', 'northwest');

  grid on;

end
