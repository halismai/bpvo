set term postscript eps enhanced color
set output "01.eps"
set size ratio -1
set xrange [-91:1918]
set yrange [-1562:447]
set xlabel "x [m]"
set ylabel "z [m]"
plot "01.txt" using 1:2 lc rgb "#FF0000" title 'Ground Truth' w lines,"01.txt" using 3:4 lc rgb "#0000FF" title 'Visual Odometry' w lines,"< head -1 01.txt" using 1:2 lc rgb "#000000" pt 4 ps 1 lw 2 title 'Sequence Start' w points
