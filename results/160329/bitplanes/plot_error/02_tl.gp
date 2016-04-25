set term postscript eps enhanced color
set output "02_tl.eps"
set size ratio 0.5
set yrange [0:*]
set xlabel "Path Length [m]"
set ylabel "Translation Error [%]"
plot "02_tl.txt" using 1:($2*100) title 'Translation Error' lc rgb "#0000FF" pt 4 w linespoints
