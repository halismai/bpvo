function B = bitplanes(I, sigma_bp, sigma_ct)
  % function B = bitplanes(I, sigma_bp, sigma_ct)
  %
  % INPUT
  %     I the input image
  %     sigma_bp smoothing on bitplanes
  %     sigma_ct smoothing on census


  C = uint32( censusTrasform(I, sigma_ct) );

  if sigma_bp > 0
    %h = fspecial('gaussian', floor(6*sigma_bp), sigma_bp);
    h = fspecial('gaussian', [5 5], sigma_bp);
  end

  for i = 8 : -1 : 1
    bb = double(bitshift(bitand(C, 2^(i-1)), -(i-1)));
    if sigma_bp > 0
      B(:,:,i) = imfilter(bb, h);
    else
      B(:,:,i) = bb;
    end
  end

end


function D = censusTrasform(I, sigma)

  if sigma > 0.0
    %I = imfilter(I, fspecial('gaussian', 1 + floor(6*sigma), sigma));
    fspecial('gaussian', 3, sigma)
    I = imfilter(I, fspecial('gaussian', 3, sigma));
  end

  C = I(2:end-1, 2:end-1);
  D = zeros(size(I));
  D(2:end-1,2:end-1) = ...
    (C >= I(1:end-2, 1:end-2)) .* 1 + ...
    (C >= I(1:end-2, 2:end-1)) .* 2 + ...
    (C >= I(1:end-2, 3:end  )) .* 4 + ...
    (C >= I(2:end-1, 1:end-2)) .* 8 + ...
    (C >= I(2:end-1, 3:end  )) .* 16 + ...
    (C >= I(3:end,   1:end-2)) .* 32 + ...
    (C >= I(3:end,   2:end-1)) .* 64 + ...
    (C >= I(3:end,   3:end  )) .* 128;

end
