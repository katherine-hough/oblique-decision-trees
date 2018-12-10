import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

/* Tracks the passage of time. */
public class Timer {

  private long startTime;

  public Timer() {
    start();
  }
  
  /* Restarts the timer */
  public void start() {
    startTime = System.currentTimeMillis();
  }

  /* Prints the amount of time that has passed since this timer was started */
  public void printElapsedTime() {
    Date date = new Date(System.currentTimeMillis() - startTime);
    SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss.SSS");
    formatter.setTimeZone(TimeZone.getTimeZone("UTC"));
    String formatted = formatter.format(date);
    System.out.printf("Elapsed Time: %s\n", formatted);
  }

  /* Prints the amount of time that has passed since this timer was started with
   * the specified message */
  public void printElapsedTime(String message) {
    Date date = new Date(System.currentTimeMillis() - startTime);
    SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss.SSS");
    formatter.setTimeZone(TimeZone.getTimeZone("UTC"));
    String formatted = formatter.format(date);
    System.out.printf("Elapsed Time: %s | %s\n", formatted, message);
  }

}
