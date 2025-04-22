all: Ex2.1 Ex2.2 Ex2.3 Ex2.4 Ex2.5 Ex2.6 Ex2.7 Ex2.8 Ex2.9 Ex2.10 Ex2.11 Ex2.13# List here all examples you completed

Ex2.1:
	@echo "Exercise 2.1"	
	python3 ./triangler.py

Ex2.2:
	@echo "Exercise 2.2"
	python3 ./triangler.py inspect --point 0.1 0.2 0.3

Ex2.3:
	@echo "Exercise 2.3"
	python3 ./triangler.py inspect --point 0.1 0.2 0.3

Ex2.4:
	@echo "Exercise 2.4"
	python3 ./triangler.py inspect --point 0.1 0.2 0.3

Ex2.5:
	@echo "Exercise 2.5"
	python3 ./triangler.py -param cartesian inspect --x_space -p 0.1 0.2 0.3
	python3 ./triangler.py -param spherical inspect --x_space -p 0.1 0.2 0.3

Ex2.6: Ex2.6a Ex2.6b
	
Ex2.6a:
	@echo "Exercise 2.6a"
	python3 ./triangler.py plot --xs 1 2 --range -0.2 0.2 --fixed_x 0.0
	python3 ./triangler.py plot --xs 0 2 --range -0.2 0.2 --fixed_x 0.0
	python3 ./triangler.py plot --xs 0 1 --range -0.2 0.2 --fixed_x 0.0

Ex2.6b:
	@echo "Exercise 2.6b"
	python3 ./triangler.py plot --x_space --xs 1 2
	python3 ./triangler.py plot --x_space --xs 0 2
	python3 ./triangler.py plot --x_space --xs 0 1

Ex2.7: Ex2.7a Ex2.7b

Ex2.7a:
	@echo "Exercise 2.7a"
	python3 ./triangler.py plot --3D --xs 1 2
	python3 ./triangler.py plot --3D --xs 0 2
	python3 ./triangler.py plot --3D --xs 0 1

Ex2.7b:
	@echo "Exercise 2.7b"
	python3 ./triangler.py plot --3D --x_space --xs 1 2
	python3 ./triangler.py plot --3D --x_space --xs 0 2
	python3 ./triangler.py plot --3D --x_space --xs 0 1

Ex2.8: Ex2.8c Ex2.8s

Ex2.8c:
	@echo "Exercise 2.8 Cartesian"
	python3 ./triangler.py -param cartesian integrate -n 10 -ppi 10000 -it naive -nc 1 -s 1337

Ex2.8s:
	@echo "Exercise 2.8 Spherical"
	python3 ./triangler.py -param spherical integrate -n 10 -ppi 10000 -it naive -nc 1 -s 1337

Ex2.9:
	@echo "Exercise 2.9"
	python3 ./triangler.py -param spherical integrate -n 10 -ppi 10000 -it naive -nc 8 -s 1337

Ex2.10:
	@echo "Exercise 2.10"
	python3 ./triangler.py -param spherical integrate -n 10 -ppi 10000 -it vegas -nc 8 -s 1337

Ex2.11: Ex2.11n Ex2.11v


Ex2.11n:
	@echo "Exercise 2.11 naive"
	python3 ./triangler.py -param spherical -mc TRUE integrate -n 10 -ppi 10000 -it naive -nc 8 -s 1337

Ex2.11v:
	@echo "Exercise 2.11 vegas"
	python3 ./triangler.py -param spherical -mc TRUE integrate -n 10 -ppi 10000 -it vegas -nc 8 -s 1337

Ex2.13n:
	@echo "Zero fermion mass integral without multi-channeling, without improved LTD"
	python3 triangler.py -param spherical --m_s 1. --m_psi 0. -p 0. 0. 0. 5. -q 1. 4. 3. 2. integrate -n 10 -ppi 10000 -it naive -nc 1 -s 1337

Ex2.13m:
	@echo "Zero fermion mass integral with multi-channeling, without improved LTD"
	python3 triangler.py --alpha 1. -mc True -param spherical --m_s 1. --m_psi 0. -p 0. 0. 0. 5. -q 1. 4. 3. 2. integrate -n 10 -ppi 10000 -it naive -nc 1 -s 1337

Ex2.13v:
	@echo "Zero fermion mass integral without multi-channeling, with improved LTD"
	python3 triangler.py --alpha 1.5 -mc True -param spherical --m_s 1. --m_psi 0. -p 0. 0. 0. 5. -q 1. 4. 3. 2. integrate -n 10 -ppi 10000 -it vegas -nc 1 -s 1337

MOMTROP_TEST:
	@echo "Surely nothing will go wrong on the first attempt"
	python3 ./triangler.py -param momtrop integrate -n 10 -ppi 100000 -it naive -s 1337

COMPARE:
	@echo "Compare all the implemented samplers for the massive/massless triangle/f=1 integrand"
	python3 ./triangler.py compare_sampl -ppi 10000 -nt 10 50 100 500