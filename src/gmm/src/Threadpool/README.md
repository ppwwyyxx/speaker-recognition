ThreadPool
==========

A simple C++11 Thread Pool implementation, providing an optional task priority.

Fork from [Jakob Progsch' repository](https://github.com/progschj/ThreadPool).


Possible improvements
---------------------

* variadic enqueue, no need for std::bind
* prevent starvation, priority aging
