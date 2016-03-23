function out = cat_struct_fields(out, b)

  fnames = fieldnames(out);

  for i = 1 : length(fnames)
    out.(fnames{i}) = [out.(fnames{i}) b.(fnames{i})];
  end

end
