

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
  
  <h3>We will dive into few fundamental concepts before discussing Joint probability</h3>
  <ul>
    <li>
      <h4>Multiplicative Probability (AND)</h4>
      <ul>
        <li>2 or more Independent Events occurs simultaneously: P(A and B) = P(A) x P(B) 
          <ul>
          <li>Eg., Flipping a Fair Coin simultaneously and getting "HEADS": <br>
            P(Heads on Coin-1 and Coin-2)= P(Head on Coin-1) * P(Head on Coin-2)= (1/2) x (1/2)=0.25 (25% chance)
          </li>
        </ul>
        </li>
      </ul>
    </li>  
    <li>
      <h4>Additive Probability (OR) </h4>
    <ul>
      <li>Mutually Exclusive Events: Probability(A or B) = Probability(A) + Probability(B) 
        <ul>
          <li>Eg., Rolling a Die and find P(2) or P(4) = P(2 or 4) = P(2) + P(4) = (1/6)+(1/6)=0.33 (33% chance)
        </ul>
      </li>
      <li>If Events are <b>not</b> Mutually Exclusive : Probability(A or B) = Probability(A) + Probability(B) - P( A and B) 
      </li>
    </ul>
    </li>
  </ul>
  <h3>Joint Probability</h3>
  
  
