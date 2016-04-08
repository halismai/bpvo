function X = transform_points(T, X)
  X = bsxfun(@plus, T(1:3,end), T(1:3,1:3)*X(1:3,:));
end
