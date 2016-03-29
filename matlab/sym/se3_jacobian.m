syms X Y Z        real
syms fx fy cx cy  real;
syms s c1 c2 c3   real;
syms Ix Iy real;

T = [...
  s 0 0 -s*c1; ...
  0 s 0 -s*c2; ...
  0 0 s -s*c3; ...
  0 0 0 1];

%T = eye(4);
V = se3_generators;
P = [X; Y; Z; 1];

Jh = [];
for i = 1 : length(V)
  Jh = [Jh inv(T)*V{i}*T*P];
end


K = [fx 0 cx; 0 fy cy; 0 0 1];
x = K * P(1:3,:);
x = x / x(end);
Jp = simplify(jacobian(x, P(1:3)));

Jh(end,:) = [];
J = simplify([Ix Iy 0] * Jp * Jh);

