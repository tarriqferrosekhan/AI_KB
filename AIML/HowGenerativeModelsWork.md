

<h1>How Generative Model Works</h1>
Reference: Generative Versus Discriminative Models Topic from the book <a href="https://www.oreilly.com/library/view/deep-learning/9781491924570">Deep Learning Book</a> by Josh Patterson, Adam Gibson
<br><br>
<table>
  <tr>
    <td><b>Attribute</b></td><td><b>Generative Models</b></td><td><b>Discriminative Model</b></td>
  </tr>
  <tr>
    <td>Focus</td>
    <td>Tries to understand data and generate a likely output given an input.</td>
    <td>Provides classification or category for a given input</td>
  </tr>
   <tr>
    <td>How it learns</td>
    <td> <a href="https://en.wikipedia.org/wiki/Joint_probability_distribution">JOINT Probability distribution</a> <b>p(x, y)</b></td>
    <td> <a href="https://en.wikipedia.org/wiki/Conditional_probability_distribution">Conditional Probability</a> <b>p(x|y)</b></td>
  </tr>
  </table>
  <hr>
  <h3>We will quickly see few fundamental concepts before discussing Joint probability</h3>
  <ul>
    <li>
      <h4>Multiplicative Probability P(A and B)</h4>
      <ul>
        <li>2 or more Independent Events occurs simultaneously, <b>Formula: P(A and B) = P(A) x P(B)</b> 
          <ul>
          <li>Eg., Flipping a Fair Coin simultaneously and getting "HEADS": <br>
            P(Heads on Coin-1 and Coin-2)= P(Head on Coin-1) * P(Head on Coin-2)= (1/2) x (1/2)=0.25 (25% chance)
          </li>
        </ul>
        </li>
      </ul>
    </li>  
    <li>
      <h4>Additive Probability P (A OR B) </h4>
    <ul>
      <li>Mutually Exclusive Events, <b>Formula: P(A or B) = P(A) + P(B) </b> 
        <ul>
          <li>Eg., Rolling a Die and find P(2) or P(4) = P(2 or 4) = P(2) + P(4) = (1/6)+(1/6)=0.33 (33% chance)
        </ul>
      </li>
      <li>If Events are <b>not</b> Mutually Exclusive, <b>Formula : P(A or B) = P(A) + P(B) - P( A and B)</b> 
      </li>
    </ul>
    </li>
    <li>
  <h4>Conditional Probability (Given that...)</h4>
  <ul>
    <li>Definition: The probability of Event A occurring given that Event B has already occurred.  
      <ul>
        <li><b>Formula: P(A | B) = P(A and B) / P(B)</b></li>
      </ul>
    </li>
    <li>Example: Suppose 30% of employees know Python, and 15% know both Python and SQL.  
      <ul>
        <li>P(Python | SQL) = P(Python and SQL) / P(SQL)</li>
        <li>If P(SQL) = 0.25, then P(Python | SQL) = 0.15 / 0.25 = 0.6 (60% chance)</li>
      </ul>
    </li>
  </ul>
      
</li>
<li>
  <h4>Bayes' Theorem (Based on Prior Knowledge)</h4>
  <ul>
    <li>Definition: Used to find the probability of an event based on prior knowledge of related conditions.  
      <ul>
        <li><b>Formula: P(A | B) = [ P(B | A) × P(A) ] / P(B)</b></li>
      </ul>
    </li>
    <li>Example: Suppose 5% of patients have a disease, and a test correctly detects it 90% of the time.  
      <ul>
        <li>P(Disease) = 0.05, P(Positive | Disease) = 0.9, P(Positive) = 0.1</li>
        <li>Then, P(Disease | Positive) = (0.9 × 0.05) / 0.1 = 0.45 (45% chance patient actually has the disease)</li>
      </ul>
    </li>
  </ul>
</li>
  </ul>
  <hr>
  <h3><u>Joint Probability</u></h3><br>
 <ul>
    <li>Definition: The probability of two events A and B occurring together.
      <ul>
        <li>Formula (General): P(A and B) = P(A) × P(B | A)</li>
      </ul>
    </li>
    <li><b>Case 1: With Replacement (Independent Events)</b>
      <ul>
        <li>After the first draw, the card is replaced — probabilities stay the same.</li>
        <li>Example: Drawing 2 Aces from a deck of 52 cards with replacement.</li>
        <li>P(A₁) = 4/52, P(A₂) = 4/52</li>
        <li>P(A₁ and A₂) = (4/52) × (4/52) = 1/169 ≈ 0.0059 (0.59%)</li>
      </ul>
    </li>
    <li><b>Case 2: Without Replacement (Dependent Events)</b>
      <ul>
        <li>After the first draw, the card is not replaced — probabilities change.</li>
        <li>Example: Drawing 2 Aces from a deck of 52 cards without replacement.</li>
        <li>P(A₁) = 4/52, P(A₂ | A₁) = 3/51</li>
        <li>P(A₁ and A₂) = (4/52) × (3/51) = 1/221 ≈ 0.0045 (0.45%)</li>
      </ul>
    </li>
  </ul>
</li>
</ul>
<hr>
<h3>Joint Probability in Text Generation</h3>
<ul>
  <li>
  <h4>Joint Probability in Text Generation ("I love pizza")</h4>
  <ul>
    <li>Definition: In generative models, the probability of generating a complete text sequence is the product of probabilities of each word conditioned on the previous ones.
      <ul>
        <li>Formula: P(I, love, pizza) = P(I) × P(love | I) × P(pizza | I, love)</li>
      </ul>
    </li>
    <li><b>Step 1: Assign Hypothetical Probabilities</b>
      <table>
        <tr><td>Step</td><td>Condition</td><td>Word Chosen</td><td>Probability</td><td>Hypothetical Probability Assigned</td><td>Cumulative Probability</td></tr>
        <tr><td>1</td><td>(none)</td><td>"I"</td><td>P(I)</td><td>0.5</td><td>0.5</td></tr>
        <tr><td>2</td><td>(none)</td><td>"Love"</td><td>P(Love|I)</td><td>0.6</td><td>0.5*0.6=0.3</td></tr>
        <tr><td>3</td><td>(none)</td><td>"Love"</td><td>P(Pizza|I,Love)</td><td>0.4</td><td>0.3*0.4=0.12</td></tr>
      </table>
      <b>P("I Love Pizza") = 0.12</b>
    </li>
    <li><b>Step 3: Interpretation</b>
      <ul>
        <li>The model multiplies the conditional probabilities of each word as it generates the sequence.</li>
        <li>Even if words like "love" or "pizza" are common, their combination as a specific ordered sequence has a smaller joint probability.</li>
        <li>Higher joint probability sequences are more likely to appear as generated outputs.</li>
      </ul>
    </li>
  </ul>
</li>
</ul>
  
