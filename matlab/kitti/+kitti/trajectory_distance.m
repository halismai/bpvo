function d = trajectory_distance(T)
  %function d = trajectory_distance(T)
  %
  % computes the trajecotry distance given poses

  d = zeros(1, size(T,3));

  for i = 2 : size(T,3)
    d(i) = d(i-1) + norm( T(1:3,end,i-1) - T(1:3,end,i) );
  end

end
