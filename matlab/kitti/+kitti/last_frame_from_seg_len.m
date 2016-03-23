function f_num = last_frame_from_seq_len(dists, first_frame, len)

  ind = find( dists > dists(first_frame)+len );

  if ~isempty(ind)
    f_num = ind(1);
  else
    f_num = -1;
  end

end
