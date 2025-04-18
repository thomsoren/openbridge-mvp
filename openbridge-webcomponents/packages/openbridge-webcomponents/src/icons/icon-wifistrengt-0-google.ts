import {LitElement, html, css, svg} from 'lit';
import {customElement, property} from 'lit/decorators.js';

@customElement('obi-wifistrengt-0-google')
export class ObiWifistrengt0Google extends LitElement {
  @property({type: Boolean}) useCssColor = false;

  private icon = svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3.5C7.31 3.5 3.07 5.4 0 8.48L12 20.5L24 8.48C20.93 5.4 16.69 3.5 12 3.5ZM2.92 8.57C5.51 6.58 8.67 5.5 12 5.5C15.33 5.5 18.49 6.58 21.08 8.57L12 17.67L2.92 8.57Z" fill="currentColor"/>
</svg>
`;

  private iconCss = svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 3.5C7.31 3.5 3.07 5.4 0 8.48L12 20.5L24 8.48C20.93 5.4 16.69 3.5 12 3.5ZM2.92 8.57C5.51 6.58 8.67 5.5 12 5.5C15.33 5.5 18.49 6.58 21.08 8.57L12 17.67L2.92 8.57Z" style="fill: var(--element-active-color)"/>
</svg>
`;

  override render() {
    return html`
      <div class="wrapper">${this.useCssColor ? this.iconCss : this.icon}</div>
    `;
  }

  static override styles = css`
    .wrapper {
      height: 100%;
      width: 100%;
      line-height: 0;
    }
    .wrapper > * {
      height: 100%;
      width: 100%;
    }
  `;
}

declare global {
  interface HTMLElementTagNameMap {
    'obi-wifistrengt-0-google': ObiWifistrengt0Google;
  }
}
