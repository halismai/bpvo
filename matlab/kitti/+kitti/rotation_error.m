function err = rotation_error(T)
  d = 0.5 * sum( diag(T(1:3,1:3)) - 1.0 );
  err = acos( max( min(d, 1.0), -1.0 ) );
end
