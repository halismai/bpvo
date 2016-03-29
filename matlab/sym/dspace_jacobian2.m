syms u v d w real;
syms fx fy b real;
syms s c1 c2 c3 real;
syms Ix Iy real;

G = [fx 0 0 0; 0 fy 0 0; 0 0 0 fx*b; 0 0 1 0];

T = [...
  s 0 0 -s*c1; ...
  0 s 0 -s*c2; ...
  0 0 s -s*c3; ...
  0 0 0 1];

V = se3_generators;

p = [u; v; d; w];

for i = 1 : length(V)
  J_T(:,i) = inv(T) * G * V{i} * inv(G) * T * p;
end

p = p ./ w;

J = [Ix Iy 0 0 ] * jacobian(p, [u v d w]) * J_T;
J = subs(J, w, 1); %
J = simplify(J);


