#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <algorithm>
#include <utility>
#include <functional>
#include <stdexcept>

// MODIFICATION: wrap inside namespace ThreadLib
namespace ThreadLib {

	class Threadpool;

	class Worker {
		public:
			Worker(Threadpool& s) : pool(s) { }

			void operator()();

		private:
			Threadpool& pool;
	};

	class Threadpool {
		public:
			typedef std::vector<std::thread>::size_type size_type;

			Threadpool() : Threadpool(std::max(1u, std::thread::hardware_concurrency())) { }
			Threadpool(size_type);
			~Threadpool();

			//template<class F> auto enqueue(F&& f, int priority = 0) -> std::future<decltype(std::forward<F>(f)())>;

			// add new work item to the pool
			template<class F>
				auto enqueue(F&& f, int priority) -> std::future<decltype(std::forward<F>(f)())> {
					typedef decltype(std::forward<F>(f)()) R;

					if(stop)
						throw std::runtime_error("enqueue on stopped threadpool");

					auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
					std::future<R> res = task->get_future();

					{
						std::unique_lock<std::mutex> lock(queue_mutex);
						tasks.emplace(priority, [task]{ (*task)(); });
					}

					condition.notify_one();

					return res;
				}

		private:
			friend class Worker;

			// need to keep track of threads so we can join them
			std::vector<std::thread> workers;

			typedef std::pair<int, std::function<void()>> priority_task;

			// emulate 'nice'
			struct task_comp {
				bool operator()(const priority_task& lhs, const priority_task& rhs) const {
					return lhs.first > rhs.first;
				}
			};

			// the prioritized task queue
			std::priority_queue<priority_task, std::vector<priority_task>, task_comp> tasks;

			// synchronization
			std::mutex queue_mutex;
			std::condition_variable condition;
			bool stop;
	};
}

#endif
