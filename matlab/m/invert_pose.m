function T = invert_pose(T)

  R = T(1:3,1:3);
  t = T(1:3,end);

  T = [ R' -R'*t; 0 0 0 1];

end
