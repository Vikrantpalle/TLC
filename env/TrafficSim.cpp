#include <vector>
#include <string>
#include <map>
#include <deque>
#include <time.h>	
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define DESTINATION_REWARD 5

struct Car {
	Car(std::deque<int> path) : path(path) {};
	std::deque<int> path;
	int waitingTime = 0;
};

struct street {
	int src, dest;
	bool green = false;
	int gKey, lKey;
	int totWaitingTime = 0;
	std::string name;
	std::deque<Car> trafficQ;
	std::vector<Car> waiting;
};

struct crossing {
	std::vector<int> inDeg;
	std::vector<int> outDeg;
};

class TrafficSim {
public:

	int globalKey=0;
	int streets = 0;
	int intersections = 0;
	int cars = 0;

	TrafficSim(int I, int S, int C) {
		this->intersections = I;
		this->streets = S;
		this->cars = C;
		globalKey = 0;
		streetMap.resize(streets);
		crossingMap.resize(intersections);
	}

	std::tuple<std::vector<int>, int, bool> longStep(std::vector<int> actions, int steps = 4) {

		//update signals
		for (int i = 0; i < intersections; i++) {
			for (auto it : crossingMap[i].inDeg) {
				streetMap[it].green = false;
			}
			streetMap[crossingMap[i].inDeg[actions[i]]].green = true;
		};

		int reward = 0;
		for (int step = 0; step < steps; step++) {
			reward += this->step();
		}
		bool done = true;
		std::vector<int> state(streets);
		for (int street=0; street < streets; street++) {
			state[streetMap[street].gKey] = streetMap[street].trafficQ.size();
			if (state[streetMap[street].gKey] > 0) done = false;
		}

		return { state, reward, done };
	}

	//return state,reward 
	int step() {

		int reward = 0;

		//move cars to waiting
		for (auto& st : streetMap) {
			if (st.green && !st.trafficQ.empty()) {
				Car car = st.trafficQ.front();
				st.trafficQ.pop_front();
				car.path.pop_front();
				car.waitingTime = 0;
				if (car.path.empty()) {
					reward += DESTINATION_REWARD;
				}
				else {
					streetMap[car.path.front()].waiting.push_back(car);
				}
			}
			for (auto& car : st.trafficQ) {
				car.waitingTime += 1;
				reward -= car.waitingTime;
			}
		}

		//move cars from waiting -> street
		for (auto& st : streetMap) {
			while (!st.waiting.empty()) {
				st.trafficQ.push_back(st.waiting.back());
				st.waiting.pop_back();
			}
		}



		return reward;

	};

	void addStreet(int src, int dest, std::string name) {
		street newStreet;
		newStreet.src = src;
		newStreet.dest = dest;
		newStreet.name = name;
		newStreet.gKey = globalKey;
		newStreet.lKey = crossingMap[dest].inDeg.size();
		streetMap[newStreet.gKey] = newStreet;
		crossingMap[newStreet.dest].inDeg.push_back(newStreet.gKey);
		crossingMap[newStreet.src].outDeg.push_back(newStreet.gKey);
		globalKey++;
	};

	void addCar(std::deque<int> const path) {
		Car newCar(path);
		streetMap[path.front()].trafficQ.push_back(newCar);
	};

	void addCars(int n, int maxlen) {
		for (int i = 0; i < n; i++) {
			addCar(randomWalk(maxlen, i, 0, 0.4));
		}
	}
	
	std::vector<int> reset() {
		std::vector<int> state(streets, 0);
		for (int street = 0; street < streets; street++) {
			streetMap[street].trafficQ = {};
		}
		addCars(cars, 4);
		for (int street = 0; street < streets; street++) {
			state[streetMap[street].gKey] = streetMap[street].trafficQ.size();
		}
		return state;
	}

	std::deque<int> randomWalk(int maxlen, int seed, int offset, float p_end_walk = 0.4) {
		srand(seed+offset);
		std::deque<int> walk;
		int cur = std::rand() % streets;
		walk.push_back(cur);
		crossing curCross = crossingMap[streetMap[cur].dest];

		while (float(std::rand()) / float((RAND_MAX)) > p_end_walk && walk.size() < maxlen) {
			cur = std::rand() % curCross.outDeg.size();
			walk.push_back(curCross.outDeg[cur]);
			curCross = crossingMap[streetMap[curCross.outDeg[cur]].dest];
		}

		return walk;
	}

	int action_space(int i) {
		return crossingMap[i].inDeg.size();
	}


private:

	std::vector<street> streetMap;
	std::vector<crossing> crossingMap;

};

PYBIND11_MODULE(flame, m) {

	py::class_<crossing>(m, "crossing")
		.def_readwrite("inDeg", &crossing::inDeg);

	py::class_<street>(m, "street")
		.def_readwrite("trafficQ", &street::trafficQ)
		.def_readwrite("src", &street::src)
		.def_readwrite("dest", &street::dest)
		.def_readwrite("name", &street::name);

	py::class_<Car>(m, "Car")
		.def_readwrite("path", &Car::path);

	py::class_<TrafficSim>(m, "TrafficSim")
		.def(py::init<int&, int&, int&>())
		.def("addCar", &TrafficSim::addCar)
		.def("addCars", &TrafficSim::addCars)
		.def("addStreet", &TrafficSim::addStreet)
		.def("step", &TrafficSim::longStep)
		.def("randomWalk", &TrafficSim::randomWalk)
		.def("action_space",&TrafficSim::action_space)
		.def("reset",&TrafficSim::reset)
		.def_readwrite("streets", &TrafficSim::streets);
};
