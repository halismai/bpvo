#include "bpvo/vector6.h"
#include <iostream>

using namespace bpvo;

int main()
{
  std::cout << "ImplementationType: " << Vector6::ImplementationType() << std::endl;
  std::cout << Vector6::Random() << std::endl;

  Vector6 a{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Vector6 b{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  std::cout << (a + b) << std::endl;
  std::cout << (2*a) << std::endl;
  std::cout << (a - b) << std::endl;
  std::cout << (a / b) << std::endl;

  a *= 2;
  std::cout << a << std::endl;

  std::cout << -a << std::endl;
  std::cout << (-1.0 * (-a)) << std::endl;
  std::cout << 1.0 / a << std::endl;

  return 0;
}
