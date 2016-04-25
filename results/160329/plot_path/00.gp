set term postscript eps enhanced color
set output "00.eps"
set size ratio -1
set xrange [-326:331]
set yrange [-86:571]
set xlabel "x [m]"
set ylabel "z [m]"
plot "00.txt" using 1:2 lc rgb "#FF0000" title 'Ground Truth' w lines,"00.txt" using 3:4 lc rgb "#0000FF" title 'Visual Odometry' w lines,"< head -1 00.txt" using 1:2 lc rgb "#000000" pt 4 ps 1 lw 2 title 'Sequence Start' w points
