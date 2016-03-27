function p = load_bpvo_results(prefix)

  p = cell(1,4);
  p{1} = load_res_file(sprintf('%s_tl.txt', prefix));
  p{2} = load_res_file(sprintf('%s_rl.txt', prefix));
  p{3} = load_res_file(sprintf('%s_ts.txt', prefix));
  p{4} = load_res_file(sprintf('%s_rs.txt', prefix));

  %p{1}(:,2) = p{1}(:,2) * 100; % convert to %
  p{2}(:,2) = rad2deg( p{2}(:,2) );

  p{3}(:,1) = 3.6 * p{3}(:,1);
  %p{3}(:,2) = p{3}(:,2) * 100;
  p{4}(:,1) = 3.6 * p{4}(:,1);
  p{4}(:,2) = rad2deg(p{4}(:,2));

end

function p = load_res_file(fn)
  [x, y] = textread(fn,'%f %f');
  p = [x y];
end
