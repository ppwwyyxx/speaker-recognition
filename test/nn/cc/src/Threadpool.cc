#include "Threadpool/Threadpool.hpp"

namespace ThreadLib {

	void Worker::operator()() {
		while(true) {
			std::unique_lock<std::mutex> lock(pool.queue_mutex);

			while(!pool.stop && pool.tasks.empty())
				pool.condition.wait(lock);

			if(pool.stop && pool.tasks.empty())
				return;

			std::function<void()> task(pool.tasks.top().second);
			pool.tasks.pop();

			lock.unlock();

			task();
		}
	}

	// the constructor just launches some amount of workers
	Threadpool::Threadpool(Threadpool::size_type threads) : stop(false) {
		workers.reserve(threads);

		for(Threadpool::size_type i = 0; i < threads; ++i)
			workers.emplace_back(Worker(*this));
	}

	// the destructor joins all threads
	Threadpool::~Threadpool() {
		stop = true;

		condition.notify_all();

		for(Threadpool::size_type i = 0; i < workers.size(); ++i)
			workers[i].join();
	}
}
