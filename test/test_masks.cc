#include <cstdint>
#include <cstdio>
#include <bitset>
#include <iostream>

int main()
{
  std::bitset<9> b;
  b[5] = 1;

  std::cout << b << std::endl;

  return 0;
}
