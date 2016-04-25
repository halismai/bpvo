set term postscript eps enhanced color
set output "01_rl.eps"
set size ratio 0.5
set yrange [0:*]
set xlabel "Path Length [m]"
set ylabel "Rotation Error [deg/m]"
plot "01_rl.txt" using 1:($2*57.3) title 'Rotation Error' lc rgb "#0000FF" pt 4 w linespoints
