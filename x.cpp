#include <iostream>
#include <cstdlib>
#include <ctime>

int main(){
	float t = (float) std::rand() / RAND_MAX;
	std::cout << t << std::endl;
}