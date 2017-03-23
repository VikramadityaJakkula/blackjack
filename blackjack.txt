;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Blackjack.lisp                                                                                               ;;
;;  A Neural Network Implementation of BlackJack Game with a reinforcement learning component                   ;;
;;  Student:Master Vikram Aditya Reddy Jakkula                                                 ;;
;;                                                                                             ;;
;;  Professor:Dr Diane Joyce Cook Holder                                                                                  ;;  
;;  Year: December 2005 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  This is the code that learns how to play blackjack, by playing it with itself                               ;;                
;;  This code implements a bonafide neural network. The network is made of a single neuron                     ;;         
;;  There is also a reward and lose mechanism to aid the learning adding the reinforcement component            ;; 
;;  The program learns with two players, one very cautious and the other one a risk taker                       ;; 
;;  That is, the initial values of the threshold are deliberately set to be far from optimal, which causes      ;;
;;  one player to stay even with a low score, and the other one to overdraw. Therefore, at the beginning,       ;;
;;  one of the players has typically a winning streak. As they play against each other, they                    ;;
;;  learn when to draw and when to stay and end up playing as peers,winning on average every other game.        ;;
;;                                                                                                              ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Premise of the game                                                                                                                 ;;
;;	The basic premise of the game is that you want to have a hand value that is closer to 21 than that of the dealer              ;;
;;	without going over 21. Other players at the table are of no concern. Your hand is strictly played out against                 ;;
;;	the hand of the dealer. The rules of play for the dealer are strictly dictated, leaving no decisions up to thedealer.         ;;
;;	Therefore, there is not a problem with the dealer or any of the other players at the table seeing the cards in your hand.     ;;
;;	Indeed, if you're playing at a shoe game, the player cards are all dealt face up.                                             ;;
;;                                                                                                                                    ;;
;;Values of the cards                                                                                                                 ;;
;;	An Ace can count as  1 or 11, as demonstrated below.                                                                          ;;
;;	The cards from 2 through with a stand 9 are valued as indicated.                                                              ;;
;;	The 10, Jack, Queen, and King are all valued at 10.                                                                           ;;
;;      The suits of the cards do not have any meaning in the game.                                                                   ;;
;;      The value of a hand is simply the sum of the point counts of each card in the hand.                                           ;;
;;                                                                                                                                    ;;
;;Example GamePlay                                                                                                                    ;;
;;	For example, a hand containing (5,7,9) has the value of 21.                                                                   ;;
;;	The Ace can be counted as  1 or 11. You need not specify which value the Ace has.                                             ;;
;;	It's assumed to always have the value that makes the best hand.                                                               ;;
;; 	An example will illustrate: Suppose that you have the beginning hand (Ace, 6). This hand can be  7 or 17.                     ;; 
;;	If you stop there, it will be 17. Let's assume that you draw another card to the hand and now have (Ace, 6, 3).               ;;
;; 	Your total hand is now 20, counting the Ace as 11.                                                                            ;;
;; 	Let's backtrack and assume that you had instead drawn a third card which was an 8.                                            ;;
;;	The hand is now (Ace, 6, 8) which totals 15. Notice that now the Ace must be counted as only 1 to avoid going over 21.        ;;
;;                                                                                                                                    ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Game Play Details                                             ;;
;;                                                               ;;
;;    The Game has 3 main options 1:Play 2:learn 3:Quit          ;;
;;                                                               ;;
;;     Play: To Play the game                                    ;;
;;     learn: Agent learns                                       ;;
;;     Quit: the Game                                       ;;
;;                                                               ;;
;;  It is advisable to learn and then play                       ;;
;;You need to enter numericals as choice For Example: 1 to Play  ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                                                               ;;
;; Action Details                                                ;; 
;;                                                               ;;
;;  Hit:     Request for Card                            ;;
;;  Stand:   Request for Cards               ;;
;;                                                               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;                   





;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                        ;;
;; Variable Declaration   ;;
;;                        ;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;; define the weights at random w0-w20
(setf *weights* '(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)) 

;;set the reward
(setf *reward* 1)
;;set the loss
(setf *lose* 1)

;; Win Loss and Tie count
(setf *won* 0)
(setf *lost* 0)
(setf *tied* 0)

;; Cards for the game
(setf *cards* '(1 2 3 4 5 6 7 8 9 10 11 12 13))

;;General use variables
(setf i 0)
(Setf j 0)
(setf temp 0)
(Setf n 0)

;; Dealer and Player Hand held cards list,score,stand and old score
(setf *episode* 0)
(setf *dealerhand* '())
(setf *playerhand* '())
(setf *dealerscore* 0)
(setf *playerscore* 0)
(setf *dealerstand* 0)
(setf *playerstand* 0)
(setf *dealeroldscore* 0)
(setf *playeroldscore* 0)

;;learning rate
(setf *alpha* 0.5)	
(setf *learncount* 0)
;;Limit Rate to getdecide the episodes
(setf *limit* 1000)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                           ;;
;; Function Declaration      ;;
;;                           ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun randint (n)
  "Calls the random-integer function from random "
  (random  n))

(defun shuffle (card)
 "Returns a randomly permuted copy of cards."
  (let ((result nil))
    (do ()
        ((null card) result)
      	(let* ((which (randint (length card)))
        (it (nth which card)))
        (push it result)
        (setq card (remove it card :count 1))))))

(defun hit(hand)
     "Returns The randomly generated card from the set of cards"
        (shuffle *cards*)
	(setf x (nth (random (length *cards*)) *cards*))
	(cond ((> x 10)(setf x 10)))
	(setf hand (cons x hand)))

(defun guess()
      "To randomly hit or stand"
       (setf x (random 2))
       (if (>= x 0)
	'hit
	'stand))

(defun refresh()
      "Resets the Variables to nil"
        (setf *dealerhand* nil)
	(setf *playerhand* nil)
	(setf *dealerscore* 0)
	(setf *playerscore* 0)
	(setf *dealerstand* 0)
	(setf *playerstand* 0)
	(setf *won* 0)
	(setf *lost* 0)
	(setf *tied* 0)
	(setf *episode* 0))

(defun aceheld(card)
"To see the existance of Ace in Set of Card"
	(cond ((null card) nil)
	      ((= 1 (first card)) t)
	      (t (aceheld (rest card)))))

(defun sum(hand)
	"count the total number of cards in hand"
		(cond ((null hand)0)
	      		(t(+ 1 (sum (rest hand))))))

(defun counthand(hand)
     "Counts the sum of total no of cards held in hand"	
	(add hand 0))

(defun add(hand score)
      "It is called in counthand to perform addition"
	(cond 	((null hand)0)
		(t (setf score (+ (first hand) (add (rest hand) score))))))

(defun getdecide(sum-of-hand)
     "Makes the decision as to hit or stand"	
	(cross (- sum-of-hand 1)))

(defun cross (sum-of-hand)
     "function called getdecide"
	(decide (* (nth sum-of-hand *weights*) 1)))

(defun decide (x)
     "Sends hit or count"
	(if (>= x 0)
	'hit
	'stand))

(defun learns (action sum-of-hand)
    "modify the NN input weights through simple method"
	(cond	((null action) nil)
		((eq action 'reward) (learn-win sum-of-hand))
		((eq action 'lose) (learn-lose sum-of-hand))))

(defun learn-win (sum)
   "increses weights for selected input"
	(setf (nth (- sum 1) *weights*) (+ (nth (- sum 1) *weights*)  0.015)))

(defun learn-lose (sum)
   "decreases weights for selected input"
	(setf (nth (- sum 1) *weights*) (- (nth (- sum 1) *weights*)  0.01)))	

(defun relearns (action sum-of-hand)
 "modify the NN input weights through the reward mechanism"
	(cond	((null action) nil)
		((eq action 'reward) (relearn-win sum-of-hand))
		((eq action 'lose) (relearn-lose sum-of-hand))))

(defun relearn-win (sum)
   "increses weights for selected input in a win situation with reward"
	(setf (nth (- sum 1) *weights*) (+ (* (nth (- sum 1) *weights*) *alpha*) *reward*)))

(defun relearn-lose (sum)
    "decreases weights for selected input in a lose situation with lose"
	(setf (nth (- sum 1) *weights*) (- (* (nth (- sum 1) *weights*) *alpha*) *lose*)))

(defun start()
    "Start function for the game"
	(format t "Welcome to BlackJack~%") 
	(welcome))

(defun welcome()
      "Welcomes and displays game menu"
	(format t "Do you want Dealer to 1=Play 2=learn 3=quit~%")
	(setf *response0* (read))
	(cond ((= *response0* 1) (refresh)(play) )
      		((= *response0* 2) (cond ((= *learncount* 0) (learnc))(t(format t "Sorry learning is performed already~%")(welcome))))
      		(t (format t "Thank you for Playing Blackjack~%")(quit))))

(defun learnc()
     "learn choice menu"
	(format t "***********************************************************************************************~%")
        (format t "Note: If the Episode Limit is modified greater than 500 then it would result in stack overflow.~%")
	(format t "We can choose one learning method at a time as the stack overflow occurs~%")
        (format t "We had set the episode limit to 1000 to actually see the performance variations by better learning~%")
        (format t "***************************************************************************************************~%")
        (format t "Please make one choice only~%")
        (format t "Do you want Dealer to 1=learn or 2=learn with reward mechanism 3=quit~%")
	(setf *response4* (read))
	(cond ((= *response4* 1) (refresh)(setf *learncount* 1)(learn)(welcome))
      		((= *response4* 2) (refresh)(Setf *learncount* 1)(reinf)(welcome))
      		(t (welcome))))

(defun play()
   " This function is used to generate the play between the dealer(computer) and player(user)"
	(cond ((= 0 (sum *dealerhand*))(setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerscore* (counthand *dealerhand*))
				   (cond ((and(= 11 *dealerscore*)(aceheld *dealerhand*))
						(format t "Dealer dealt Black Jack!!~%")
						(setf *dealerstand* 1)))))
	(cond ((= 0 (sum *playerhand*))(setf *playerhand* (hit *playerhand*))
				   (setf *playerhand* (hit *playerhand*))
				   (setf *playerscore* (counthand *playerhand*))
				   (cond ((and(= 11 *playerscore*)(aceheld *playerhand*))
						(format t "Player dealt Black Jack!!~%")
						(setf *playerstand* 1)))))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond 	((> *dealerstand* 0)(format t "Dealer through the Game as stand~%~%"))
		((< *dealerscore* 21)(format t "dealer's hand is ~A~%" *dealerhand*)
				 (cond ((aceheld *dealerhand*)
					(format t "with a score of ~A or ~A~%" *dealerscore* (+ *dealerscore* 10)))
				       (t (format t "with a score of ~A~%" *dealerscore*)))
				 (setf *dealeroldscore* *dealerscore*)
				 (princ "Dealer would you like to 1=Hit or 2=Stand")
				 (setf response (getdecide *dealerscore*))
				 (cond ((eq response 'hit)(format t "Dealer chose to hit~%")
						      (setf *dealerhand* (hit *dealerhand*))
						      (format t "Dealer got a ~A~%~%" (first *dealerhand*)))
				       (t(format t "Dealer chose to stand~%~%")
					(setf *dealerstand* 1)))))
	(cond 	((> *playerstand* 0)(format t "player is through the game as stand~%~%"))
		((< *playerscore* 21)(format t "player's hand is ~A~%" *playerhand*)
				 (cond ((aceheld *playerhand*)
					(format t "with a score of ~A or ~A~%" *playerscore* (+ *playerscore* 10)))
				       (t (format t "with a score of ~A~%" *playerscore*)))
				 (setf *playeroldscore* *playerscore*)
				 (princ "Player would you like to 1=Hit or 2=Stand")
				 (setf response3 (read))                                         
				 (cond ((or (eq response3 'hit) (= response3 '1))(format t "Player chose to hit~%")
						      (setf *playerhand* (hit *playerhand*))
						      (format t "You got a ~A~%~%" (first *playerhand*)))
				       (t(format t "Player chose to stand~%~%")(setf *playerstand* 1)))))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond ((or(and(< *dealerscore* 21)(= *dealerstand* 0))(and(< *playerscore* 21)(= *playerstand* 0)))(play))      
	      (t(cond ((and(< *dealerscore* 12)(aceheld *dealerhand*))(setf *dealerscore* (+ *dealerscore* 10))))
		(cond ((and(< *playerscore* 12)(aceheld *playerhand*))(setf *playerscore* (+ *playerscore* 10))))
		(cond ((> *dealerscore* 21)(format t "Dealer loses!!~%"))
		      ((= *dealerscore* 21)(format t "Dealer has 21!!~%")))
		(cond ((> *playerscore* 21) (format t "Player  loses!!~%"))
		      ((= *playerscore* 21)(format t "Player has 21!!~%")))
		(format t "Dealer score was ~A with a hand of ~A~%~%" *dealerscore* *dealerhand*)
		(format t "Player score was ~A with a hand of ~A~%~%" *playerscore* *playerhand*)
		(cond ((and (> *dealerscore* 21)(> *playerscore* 21))(format t "Both lose!~%"))
		      ((or(and (> *dealerscore* *playerscore*)(< *dealerscore* 22))
			  (and (< *dealerscore* 22)(> *playerscore* 21)))(format t "Dealer wins!~%"))
		      ((or(and (> *playerscore* *dealerscore*)(< *playerscore* 22))
			  (and (< *playerscore* 22)(> *dealerscore* 21)))(format t "Player wins!~%"))
		      (t (format t "Dealer Player tied!!")))
                (princ "Would you like to play again (1=yes 2=no)")
		(setf response2 (read))
		(cond ((= response2 1)(refresh)(play)) 
                      (t (welcome))))))

(defun learn()
     "This does the learning through normal simple method against a random move generating player"
	(cond ((= 0 (sum *dealerhand*))(setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerscore* (counthand *dealerhand*))
				   (cond ((and(= 11 *dealerscore*)(aceheld *dealerhand*))
						(format t "Dealer dealt Black Jack!!~%")
						(setf *dealerstand* 1)))	))
	(cond ((= 0 (sum *playerhand*))(setf *playerhand* (hit *playerhand*))
				   (setf *playerhand* (hit *playerhand*))
				   (setf *playerscore* (counthand *playerhand*))
				   (cond ((and(= 11 *playerscore*)(aceheld *playerhand*))
						(format t "Player dealt Black Jack!!~%")
						(setf *playerstand* 1)))	))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond 	((> *dealerstand* 0)(format t "Dealer through with game as stand~%~%"))
		((< *dealerscore* 21)(format t "Dealers hand is ~A~%" *dealerhand*)
				 (cond ((aceheld *dealerhand*)
					(format t "with a score of ~A or ~A~%" *dealerscore* (+ *dealerscore* 10))
						(cond ((< (counthand *dealerhand*) 22) (learns 'reward *dealerscore*))))
				       (t (format t "with a score of ~A~%" *dealerscore*)))
				 (setf *dealeroldscore* *dealerscore*)
				 (princ "Dealer would you like to 1=Hit or 2=Stand")
				 (setf response (getdecide *dealerscore*))
				 (cond ((eq response 'hit)(format t "Dealer chose to hit~%")
						      (setf *dealerhand* (hit *dealerhand*))
						      (format t "You got a ~A~%~%" (first *dealerhand*)))
				       (t(format t "Dealer chose to stand~%~%") 
					 (setf *dealerstand* 1)))))
	(cond 	((> *playerstand* 0)(format t "player through with game as stand~%~%"))
		((< *playerscore* 21)(format t "players hand is ~A~%" *playerhand*)
				 (cond ((aceheld *playerhand*)
					(format t "with a score of ~A or ~A~%" *playerscore* (+ *playerscore* 10))
						(cond ((< (counthand *playerhand*) 22) (learns 'reward *playerscore*))))
				       (t (format t "with a score of ~A~%" *playerscore*)))
				 (setf *playeroldscore* *playerscore*)
				 (princ "Player would you like to 1=Hit or 2=Stand")
				 (setf response3 (guess))
				 (cond ((eq response3 'hit)(format t "Player chose to hit~%")
						      (setf *playerhand* (hit *playerhand*))
						      (format t "Player got a ~A~%~%" (first *playerhand*)))
				       (t(format t "Player chose to stand~%~%")(setf *playerstand* 1)))))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond ((or(and(< *dealerscore* 21)(= *dealerstand* 0))(and(< *playerscore* 21)(= *playerstand* 0)))(learn))	      
	      (t(cond ((and(< *dealerscore* 12)(aceheld *dealerhand*))(setf *dealerscore* (+ *dealerscore* 10))))
		(cond ((and(< *playerscore* 12)(aceheld *playerhand*))(setf *playerscore* (+ *playerscore* 10))))
		(cond ((> *dealerscore* 21)(format t "Dealer loses!!~%") 
					(learns 'lose *dealeroldscore*))
		      ((= *dealerscore* 21)(format t "Dealer has 21!!~%")))
		(cond ((> *playerscore* 21) (format t "Player loses!!~%")
					(learns 'lose *playeroldscore*))
		      ((= *playerscore* 21)(format t "Player has 21!!~%")))
		(format t "Dealers score was ~A with a hand of ~A~%~%" *dealerscore* *dealerhand*)
		(format t "players score was ~A with a hand of ~A~%~%" *playerscore* *playerhand*)
		(cond ((and (> *dealerscore* 21)(> *playerscore* 21))(format t "You both lose!~%")(setf *lost* (+ *lost* 1)))
		      ((or(and (> *dealerscore* *playerscore*)(< *dealerscore* 22))
			  (and (< *dealerscore* 22)(> *playerscore* 21)))(format t "dealer wins!~%")(setf *won* (+ *won* 1)))
		      ((or(and (> *playerscore* *dealerscore*)(< *playerscore* 22))
			  (and (< *playerscore* 22)(> *dealerscore* 21)))(format t "player wins!~%")(setf *lost* (+ *lost* 1)))
		      (t (format t "Dealer player tied!!")(setf *tied* (+ *tied* 1)) ))   
		(setf *dealerhand* nil)(setf *playerhand* nil)(setf *dealerscore* 0)(setf *playerscore* 0)(setf *dealerstand* 0)(setf *playerstand* 0)
			(cond ((= *episode* 0) (setf *episode* (+ *episode* 1))
						  (format t "~A~%" *weights*)(learn))
			      ((< *episode* *limit*) (setf *episode* (+ *episode* 1))(learn) )
			      (t (format t "~A~%" *weights*) (format t "learning stopped after ~A games~%" *limit*)(format t "Won ~A~%" *won*)(format t "Lost ~A~%" *lost*)(format t "tied ~A~%" *tied*) (welcome))) )))

(defun reinf()
      "This learns based on the reward mechanism against the player who follows the neural weights for his decisions"
	(cond ((= 0 (sum *dealerhand*))(setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerhand* (hit *dealerhand*))
				   (setf *dealerscore* (counthand *dealerhand*))
				   (cond ((and(= 11 *dealerscore*)(aceheld *dealerhand*))
						(format t "Dealer dealt Black Jack!!~%")
						(setf *dealerstand* 1)))	))
	(cond ((= 0 (sum *playerhand*))(setf *playerhand* (hit *playerhand*))
				   (setf *playerhand* (hit *playerhand*))
				   (setf *playerscore* (counthand *playerhand*))
				   (cond ((and(= 11 *playerscore*)(aceheld *playerhand*))
						(format t "Player dealt Black Jack!!~%")
						(setf *playerstand* 1)))	))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond 	((> *dealerstand* 0)(format t "Dealer through with game as stand~%~%"))
		((< *dealerscore* 21)(format t "Dealers hand is ~A~%" *dealerhand*)
				 (cond ((aceheld *dealerhand*)
					(format t "with a score of ~A or ~A~%" *dealerscore* (+ *dealerscore* 10))
						(cond ((< (counthand *dealerhand*) 22) (relearns 'reward *dealerscore*))))
				       (t (format t "with a score of ~A~%" *dealerscore*)))
				 (setf *dealeroldscore* *dealerscore*)
				 (princ "Dealer would you like to 1=Hit or 2=Stand")
				 (setf response (getdecide *dealerscore*))
				 (cond ((eq response 'hit)(format t "Dealer chose to hit~%")
						      (setf *dealerhand* (hit *dealerhand*))
						      (format t "You got a ~A~%~%" (first *dealerhand*)))
				       (t(format t "Dealer chose to stand~%~%")
					 (setf *dealerstand* 1)))))
	(cond 	((> *playerstand* 0)(format t "player through with game as stand~%~%"))
		((< *playerscore* 21)(format t "players hand is ~A~%" *playerhand*)
				 (cond ((aceheld *playerhand*)
					(format t "with a score of ~A or ~A~%" *playerscore* (+ *playerscore* 10))
						(cond ((< (counthand *playerhand*) 22) (relearns 'reward *playerscore*))))
				       (t (format t "with a score of ~A~%" *playerscore*)))
				 (setf *playeroldscore* *playerscore*)
				 (princ "Player would you like to 1=Hit or 2=Stand")
				 (setf response3 (getdecide *playerscore*))
				 (cond ((eq response3 'hit)(format t "Player chose to hit~%")
						      (setf *playerhand* (hit *playerhand*))
						      (format t "Player got a ~A~%~%" (first *playerhand*)))
				       (t(format t "Player chose to stand~%~%")(setf *playerstand* 1)))))
	(setf *dealerscore* (counthand *dealerhand*))
	(setf *playerscore* (counthand *playerhand*))
	(cond ((or(and(< *dealerscore* 21)(= *dealerstand* 0))(and(< *playerscore* 21)(= *playerstand* 0)))(reinf))
	      (t(cond ((and(< *dealerscore* 12)(aceheld *dealerhand*))(setf *dealerscore* (+ *dealerscore* 10))))
		(cond ((and(< *playerscore* 12)(aceheld *playerhand*))(setf *playerscore* (+ *playerscore* 10))))
		(cond ((> *dealerscore* 21)(format t "Dealer loses!!~%") 
					(relearns 'lose *dealeroldscore*))
		      ((= *dealerscore* 21)(format t "Dealer has 21!!~%")))
		(cond ((> *playerscore* 21) (format t "Player loses!!~%")
					(relearns 'lose *playeroldscore*))
		      ((= *playerscore* 21)(format t "Player has 21!!~%")))
		(format t "Dealers score was ~A with a hand of ~A~%~%" *dealerscore* *dealerhand*)
		(format t "players score was ~A with a hand of ~A~%~%" *playerscore* *playerhand*)
		(cond ((and (> *dealerscore* 21)(> *playerscore* 21))(format t "You both lose!~%")(setf *lost* (+ *lost* 1)))
		      ((or(and (> *dealerscore* *playerscore*)(< *dealerscore* 22))
			  (and (< *dealerscore* 22)(> *playerscore* 21)))(format t "dealer wins!~%")(setf *won* (+ *won* 1)))
		      ((or(and (> *playerscore* *dealerscore*)(< *playerscore* 22))
			  (and (< *playerscore* 22)(> *dealerscore* 21)))(format t "player wins!~%")(setf *lost* (+ *lost* 1)))
		      (t (format t "Dealer player tied!!")(setf *tied* (+ *tied* 1)) ))   
		(setf *dealerhand* nil)(setf *playerhand* nil)(setf *dealerscore* 0)(setf *playerscore* 0)(setf *dealerstand* 0)(setf *playerstand* 0)
			(cond ((= *episode* 0) (setf *episode* (+ *episode* 1))
						  (format t "~A~%" *weights*)(reinf))
			      ((< *episode* *limit*) (setf *episode* (+ *episode* 1))(reinf))
			      (t (format t "~A~%" *weights*) (format t "learning stopped after ~A games~%" *limit*)(format t "Won ~A~%" *won*)(format t "Lost ~A~%" *lost*)(format t "tied ~A~%" *tied*) (welcome))) )))
