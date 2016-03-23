function h = plot_errors(errors, name, varargin)
  %function h = plot_errors(errors, name, varargin)
  %
  % INPUT:
  %     errors list of errors in a cell array. Each has 2 column and ordered as
  %       errors{1} [x y] translation error wrt length
  %       errors{2} ditto rotation
  %       errors{3} [x y] translation error wrt speed
  %       errors{4} ditto rotation
  %
  h = zeros(4, 1);

  t_l = errors{1}; % translation length
  r_l = errors{2}; % rotation length

  t_s = errors{3}; % translation speed
  r_s = errors{4}; % rotation speed


  subplot(2, 2, 1);
  h(1) = plot(t_l(:, 1), t_l(:, 2), varargin{:});
  xlabel('Translation Error [%]'); ylabel('Path Length [m]'); grid on; hold on;
  %legend(name);

  subplot(2, 2, 2);
  h(2) = plot(r_l(:, 1), r_l(:, 2), varargin{:});
  xlabel('Rotation Error [deg/m]'); ylabel('Path Length [m]'); grid on; hold on;
  %legend(name);

  subplot(2, 2, 3);
  h(3) = plot(t_s(:, 1), t_s(:, 2), varargin{:});
  xlabel('Translation Error [%]'); ylabel('Speed [km/h]'); grid on; hold on;
  %legend(name);

  subplot(2, 2, 4);
  h(4) = plot(r_s(:, 1), r_s(:, 2), varargin{:});
  xlabel('Rotation Error [deg/m]'); ylabel('Speed [km/h]'); grid on; hold on;
  %legend(name);

end
