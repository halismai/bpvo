syms x y d w real
syms fx fy cx cy b real
syms Ix Iy real

syms s c1 c2 c3 real;

G = [fx 0 0 0; 0 fy 0 0; 0 0 0 fx*b; 0 0 1 0];

T = [...
  s 0 0 -s*c1; ...
  0 s 0 -s*c2; ...
  0 0 s -s*c3; ...
  0 0 0 1];

T = eye(4);

V = se3_generators;

for i = 1 : length(V)
  J_T(:,i) = inv(T) * G * V{i} * inv(G) * T * [x - cx; y - cy; d; w];
end

p = ([x-cx; y-cy; d; w] ./ w) + [cx; cy; 0; 0];
J_pi = [diff(p,x), diff(p,y), diff(p,d), diff(p,w)];

J = [Ix Iy 0 0] * J_pi * J_T;
J = simplify( subs(J, w, 1) );
J = subs(subs(J, x - cx, 'u'), y - cy, 'v');

