function s = skw(w)
  %function s = skw(w)
  %
  % convert a 3 vector to a skew-symmetric matrix
  %
  % INPUT
  %    w    a 3 vector
  %
  % OUTPUT
  %    s    skew symmetric matrix formed from 'w'

  % Hatem Alismail <halismai@cs.cmu.edu>
  % Last modified: Sun 02 Jun 2013 06:46:17 PM EDT

  error( nargchk(1,1,nargin) );
  assert(numel(w) == 3, 'skw: input must be a 3 vector');

  %s = [0 w(3) -w(2);-w(3) 0 w(1);w(2) -w(1) 0];
  s = [0   -w(3) w(2); ...
       w(3)  0  -w(1); ...
      -w(2) w(1) 0];

end % skw

