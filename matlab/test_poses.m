%T = reshape(load('../build/poses.txt'), 4, 4, []);
D = load('../build/poses.txt');
T = repmat(eye(4), [1 1 size(D,1)/4]);
for i = 1 : size(D,1)/4
  is = 4*(i-1) + 1;
  ie = is + 4 - 1;
  T(:,:,i) = reshape(D(is:ie,:),4,4);
end
%for i = 1 : size(T,3), T(:,:,i) = T(:,:,i).'; end
files = glob('../build/X*.txt');


X = [];
clf;
for i = 1 : 50
  X{i} = [bsxfun(@plus, T(1:3,1:3,i) * load(files{i})', T(1:3,end))];
  plot33(X{i}); hold on; view([0 -1 0]); drawnow;
end



