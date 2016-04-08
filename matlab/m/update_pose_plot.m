function update_pose_plot(hdle, T)
  C = squeeze(T(1:3,end,1:end));
  set(hdle, 'xdata', C(1,:), 'ydata', C(2,:), 'zdata', C(3,:));
end
