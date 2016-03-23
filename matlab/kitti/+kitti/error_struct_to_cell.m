function out = error_struct_to_cell(len_err, speed_err)
  out = cell(4, 1);
  out{1} = [len_err.x' len_err.y_t'];
  out{2} = [len_err.x' len_err.y_r'];
  out{3} = [speed_err.x' speed_err.y_t'];
  out{4} = [speed_err.x' speed_err.y_r'];
end
