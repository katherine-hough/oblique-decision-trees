import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class Timer {

  private long startTime;

  public void start() {
    startTime = System.currentTimeMillis();
  }

  public void printElapsedTime() {
    Date date = new Date(System.currentTimeMillis() - startTime);
    SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss.SSS");
    formatter.setTimeZone(TimeZone.getTimeZone("UTC"));
    String formatted = formatter.format(date);
    System.out.println("Elapsed Time: " + formatted);
  }

  public void printElapsedTime(String message) {
    Date date = new Date(System.currentTimeMillis() - startTime);
    SimpleDateFormat formatter = new SimpleDateFormat("HH:mm:ss.SSS");
    formatter.setTimeZone(TimeZone.getTimeZone("UTC"));
    String formatted = formatter.format(date);
    System.out.println("Elapsed Time: " + formatted + " | " + message);
  }

}
