function do_write_result(fn, T)

  fid = fopen(fn, 'w');
  assert(fid >= 0);


  for i = 1 : size(T,3)
    fprintf(fid, '%f %f %f %f %f %f %f %f %f %f %f %f\n', ...
      T(1,1,i), T(1,2,i), T(1,3,i), T(1,4,i), ...
      T(2,1,i), T(2,2,i), T(2,3,i), T(2,4,i), ...
      T(3,1,i), T(3,2,i), T(3,3,i), T(3,4,i));

  end

  fclose(fid);

end
